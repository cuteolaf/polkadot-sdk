// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Warp sync support.

pub use sp_consensus_grandpa::{AuthorityList, SetId};

use crate::{
	chain_sync::validate_blocks,
	schema::v1::{StateRequest, StateResponse},
	state::{ImportResult, StateSync},
	types::{BadPeer, OpaqueStateRequest, OpaqueStateResponse},
};
use codec::{Decode, Encode};
use futures::channel::oneshot;
use libp2p::PeerId;
use log::{debug, error, trace};
use sc_client_api::ProofProvider;
use sc_network_common::sync::message::{
	BlockAttributes, BlockData, BlockRequest, Direction, FromBlock,
};
use sp_blockchain::HeaderBackend;
use sp_runtime::traits::{Block as BlockT, Header, NumberFor, Zero};
use std::{collections::HashMap, fmt, sync::Arc};

/// Log target for this file.
const LOG_TARGET: &'static str = "sync";

/// Scale-encoded warp sync proof response.
pub struct EncodedProof(pub Vec<u8>);

/// Warp sync request
#[derive(Encode, Decode, Debug, Clone)]
pub struct WarpProofRequest<B: BlockT> {
	/// Start collecting proofs from this block.
	pub begin: B::Hash,
}

/// Proof verification result.
pub enum VerificationResult<Block: BlockT> {
	/// Proof is valid, but the target was not reached.
	Partial(SetId, AuthorityList, Block::Hash),
	/// Target finality is proved.
	Complete(SetId, AuthorityList, Block::Header),
}

/// Warp sync backend. Handles retrieving and verifying warp sync proofs.
pub trait WarpSyncProvider<Block: BlockT>: Send + Sync {
	/// Generate proof starting at given block hash. The proof is accumulated until maximum proof
	/// size is reached.
	fn generate(
		&self,
		start: Block::Hash,
	) -> Result<EncodedProof, Box<dyn std::error::Error + Send + Sync>>;
	/// Verify warp proof against current set of authorities.
	fn verify(
		&self,
		proof: &EncodedProof,
		set_id: SetId,
		authorities: AuthorityList,
	) -> Result<VerificationResult<Block>, Box<dyn std::error::Error + Send + Sync>>;
	/// Get current list of authorities. This is supposed to be genesis authorities when starting
	/// sync.
	fn current_authorities(&self) -> AuthorityList;
}

mod rep {
	/// Unexpected response received form a peer
	pub const UNEXPECTED_RESPONSE: Rep = Rep::new(-(1 << 29), "Unexpected response");

	/// Peer provided invalid warp proof data
	pub const BAD_WARP_PROOF: Rep = Rep::new(-(1 << 29), "Bad warp proof");

	/// Peer did not provide us with advertised block data.
	pub const NO_BLOCK: Rep = Rep::new(-(1 << 29), "No requested block data");

	/// Reputation change for peers which send us non-requested block data.
	pub const NOT_REQUESTED: Rep = Rep::new(-(1 << 29), "Not requested block data");

	/// Reputation change for peers which send us a block which we fail to verify.
	pub const VERIFICATION_FAIL: Rep = Rep::new(-(1 << 29), "Block verification failed");
}

/// Reported warp sync phase.
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum WarpSyncPhase<Block: BlockT> {
	/// Waiting for peers to connect.
	AwaitingPeers { required_peers: usize },
	/// Waiting for target block to be received.
	AwaitingTargetBlock,
	/// Downloading and verifying grandpa warp proofs.
	DownloadingWarpProofs,
	/// Downloading target block.
	DownloadingTargetBlock,
	/// Downloading state data.
	DownloadingState,
	/// Importing state.
	ImportingState,
	/// Downloading block history.
	DownloadingBlocks(NumberFor<Block>),
}

impl<Block: BlockT> fmt::Display for WarpSyncPhase<Block> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::AwaitingPeers { required_peers } =>
				write!(f, "Waiting for {required_peers} peers to be connected"),
			Self::AwaitingTargetBlock => write!(f, "Waiting for target block to be received"),
			Self::DownloadingWarpProofs => write!(f, "Downloading finality proofs"),
			Self::DownloadingTargetBlock => write!(f, "Downloading target block"),
			Self::DownloadingState => write!(f, "Downloading state"),
			Self::ImportingState => write!(f, "Importing state"),
			Self::DownloadingBlocks(n) => write!(f, "Downloading block history (#{})", n),
		}
	}
}

/// Reported warp sync progress.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct WarpSyncProgress<Block: BlockT> {
	/// Estimated download percentage.
	pub phase: WarpSyncPhase<Block>,
	/// Total bytes downloaded so far.
	pub total_bytes: u64,
}

/// The different types of warp syncing, passed to `build_network`.
pub enum WarpSyncParams<Block: BlockT> {
	/// Standard warp sync for the chain.
	WithProvider(Arc<dyn WarpSyncProvider<Block>>),
	/// Skip downloading proofs and wait for a header of the state that should be downloaded.
	///
	/// It is expected that the header provider ensures that the header is trusted.
	WaitForTarget(oneshot::Receiver<<Block as BlockT>::Header>),
}

/// Warp sync configuration as accepted by [`WarpSync`].
pub enum WarpSyncConfig<Block: BlockT> {
	/// Standard warp sync for the chain.
	WithProvider(Arc<dyn WarpSyncProvider<Block>>),
	/// Skip downloading proofs and wait for a header of the state that should be downloaded.
	///
	/// It is expected that the header provider ensures that the header is trusted.
	WaitForTarget,
}

impl<Block: BlockT> WarpSyncParams<Block> {
	/// Split `WarpSyncParams` into `WarpSyncConfig` and warp sync target block header receiver.
	pub fn split(
		self,
	) -> (WarpSyncConfig<Block>, Option<oneshot::Receiver<<Block as BlockT>::Header>>) {
		match self {
			WarpSyncParams::WithProvider(provider) =>
				(WarpSyncConfig::WithProvider(provider), None),
			WarpSyncParams::WaitForTarget(rx) => (WarpSyncConfig::WaitForTarget, Some(rx)),
		}
	}
}

/// Warp sync phase.
enum Phase<B: BlockT, Client> {
	/// Waiting for enough peers to connect.
	WaitingForPeers,
	/// Downloading warp proofs.
	WarpProof {
		set_id: SetId,
		authorities: AuthorityList,
		last_hash: B::Hash,
		warp_sync_provider: Arc<dyn WarpSyncProvider<B>>,
	},
	/// Waiting for target block to be set externally if we skip warp proofs downloading,
	/// and start straight from the target block (used by parachains warp sync).
	PendingTargetBlock,
	/// Downloading target block.
	TargetBlock(B::Header),
	/// Downloading state.
	State(StateSync<B, Client>),
}

/// Import warp proof result.
pub enum WarpProofImportResult {
	/// Import was successful.
	Success,
	/// Bad proof.
	BadResponse,
}

/// Import target block result.
pub enum TargetBlockImportResult {
	/// Import was successful.
	Success,
	/// Invalid block.
	BadResponse,
}

enum PeerState {
	Available,
	DownloadingProofs,
	DownloadingTargetBlock,
	DownloadingState,
}

impl PeerState {
	fn is_available(&self) -> bool {
		matches!(self, PeerState::Available)
	}
}

struct Peer<B: BlockT> {
	best_number: NumberFor<B>,
	state: PeerState,
}

enum WarpSyncAction<B: BlockT> {
	/// Send warp proof request to peer.
	SendWarpProofRequest { peer_id: PeerId, request: WarpProofRequest<B> },
	/// Send block request to peer. Always implies dropping a stale block request to the same peer.
	SendBlockRequest { peer_id: PeerId, request: BlockRequest<B> },
	/// Send state request to peer.
	SendStateRequest { peer_id: PeerId, request: OpaqueStateRequest },
	/// Disconnect and report peer.
	DropPeer(BadPeer),
}

/// Warp sync state machine. Accumulates warp proofs and state.
pub struct WarpSync<B: BlockT, Client> {
	phase: Phase<B, Client>,
	client: Arc<Client>,
	total_proof_bytes: u64,
	peers: HashMap<PeerId, Peer<B>>,
	actions: Vec<WarpSyncAction<B>>,
}

impl<B, Client> WarpSync<B, Client>
where
	B: BlockT,
	Client: HeaderBackend<B> + ProofProvider<B> + 'static,
{
	/// Create a new instance. When passing a warp sync provider we will be checking for proof and
	/// authorities. Alternatively we can pass a target block when we want to skip downloading
	/// proofs, in this case we will continue polling until the target block is known.
	pub fn new(client: Arc<Client>, warp_sync_config: WarpSyncConfig<B>) -> Self {
		let last_hash = client.hash(Zero::zero()).unwrap().expect("Genesis header always exists");
		match warp_sync_config {
			WarpSyncConfig::WithProvider(warp_sync_provider) => {
				let phase = Phase::WarpProof {
					set_id: 0,
					authorities: warp_sync_provider.current_authorities(),
					last_hash,
					warp_sync_provider: warp_sync_provider.clone(),
				};
				Self { client, phase, total_proof_bytes: 0, peers: HashMap::new() }
			},
			WarpSyncConfig::WaitForTarget =>
				Self { client, phase: Phase::PendingTargetBlock, total_proof_bytes: 0, peers: HashMap::new() },
		}
	}

	/// Set target block externally in case we skip warp proof downloading.
	pub fn set_target_block(&mut self, header: B::Header) {
		let Phase::PendingTargetBlock = self.phase else {
			error!(
				target: LOG_TARGET,
				"Attempt to set warp sync target block in invalid phase.",
			);
			debug_assert!(false);
			return
		};

		self.phase = Phase::TargetBlock(header);
	}

	pub fn new_peer(&mut self, peer_id: PeerId, _best_hash: B::Hash, best_number: NumberFor<B>) {
		self.peers.insert(peer_id, WarpPeer { best_number, state: PeerState::Available });
	}

	pub fn peer_disconnected(&mut self, peer_id: &PeerId) {
		// TODO: update requests?
		self.peers.remove(peer_id);
	}



	///  Process warp proof response.
	pub fn on_warp_proof_response(&mut self, peer_id: &PeerId, response: EncodedProof) {
		if let Some(peer) = self.peers.get_mut(peer_id) {
			peer.state = PeerState::Available;
		}

		let Phase::WarpProof { set_id, authorities, last_hash, warp_sync_provider } = &mut self.phase else {
			debug!(target: "sync", "Unexpected warp proof response");
			self.actions.push(WarpSyncAction::DropPeer(BadPeer(*peer_id, rep::UNEXPECTED_RESPONSE)));
			return
		};

		match warp_sync_provider.verify(&response, *set_id, authorities.clone()) {
			Err(e) => {
				debug!(target: "sync", "Bad warp proof response: {}", e);
				self.actions.push(WarpSyncAction::DropPeer(BadPeer(*peer_id, rep::BAD_WARP_PROOF)))
			},
			Ok(VerificationResult::Partial(new_set_id, new_authorities, new_last_hash)) => {
				log::debug!(target: "sync", "Verified partial proof, set_id={:?}", new_set_id);
				*set_id = new_set_id;
				*authorities = new_authorities;
				*last_hash = new_last_hash;
				self.total_proof_bytes += response.0.len() as u64;
			},
			Ok(VerificationResult::Complete(new_set_id, _, header)) => {
				log::debug!(target: "sync", "Verified complete proof, set_id={:?}", new_set_id);
				self.total_proof_bytes += response.0.len() as u64;
				self.phase = Phase::TargetBlock(header);
			},
		}
	}

	/// Process (target) block response.
	pub fn on_block_response(
		&mut self,
		peer_id: PeerId,
		request: BlockRequest<B>,
		blocks: Vec<BlockData<B>>,
	) {
		if let Err(bad_peer) = self.on_block_response_inner(peer_id, request, blocks) {
			self.actions.push(WarpSyncAction::DropPeer(bad_peer));
		}
	}

	pub fn on_block_response_inner(
		&mut self,
		peer_id: PeerId,
		request: BlockRequest<B>,
		blocks: Vec<BlockData<B>>,
	) -> Result<(), BadPeer> {
		if let Some(peer) = self.peers.get_mut(&peer_id) {
			peer.state = PeerState::Available;
		}

		let Phase::TargetBlock(header) = &mut self.phase else {
			debug!(target: "sync", "Unexpected target block response from {peer_id}");
			return Err(BadPeer(peer_id, rep::UNEXPECTED_RESPONSE))
		};

		if blocks.is_empty() {
			debug!(
				target: LOG_TARGET,
				"Importing target block failed: empty block response from {peer_id}",
			);
			return Err(BadPeer(peer_id, rep::NO_BLOCK))
		}

		if blocks.len() > 1 {
			debug!(
				target: LOG_TARGET,
				"Too many blocks ({}) in warp target block response from {peer_id}",
				blocks.len(),
			);
			return Err(BadPeer(peer_id, rep::NOT_REQUESTED))
		}

		validate_blocks::<B>(&blocks, &peer_id, Some(request))?;

		let block = blocks.pop().expect("`blocks` len checked above; qed");

		let Some(block_header) = &block.header else {
			log::debug!(
				target: "sync",
				"Importing target block failed: missing header in response from {peer_id}.",
			);
			return Err(BadPeer(peer_id, rep::VERIFICATION_FAIL))
		};

		if block_header != header {
			log::debug!(
				target: "sync",
				"Importing target block failed: different header in response from {peer_id}.",
			);
			return Err(BadPeer(peer_id, rep::VERIFICATION_FAIL))
		}

		if block.body.is_none() {
			log::debug!(
				target: "sync",
				"Importing target block failed: missing body in response from {peer_id}.",
			);
			return Err(BadPeer(peer_id, rep::VERIFICATION_FAIL))
		}

		let state_sync = StateSync::new(
			self.client.clone(),
			header.clone(),
			block.body,
			block.justifications,
			false,
		);
		self.phase = Phase::State(state_sync);
		Ok(())
	}


	pub fn on_state_response(&mut self, peer_id: PeerId, response: OpaqueStateResponse) {
		if let Some(peer) = self.peers.get_mut(&peer_id) {
			peer.state = PeerState::Available;
		}

		let Phase::State(sync) = &mut self.phase else {
			log::debug!(target: "sync", "Unexpected state response");
			self.actions.push(WarpSyncAction::DropPeer(BadPeer(peer_id, rep::UNEXPECTED_RESPONSE)));
			return
		};

		// TODO: decode response.

		sync.import(response);
	}

	///  Validate and import a state response.
	pub fn import_state(&mut self, response: StateResponse) -> ImportResult<B> {
		match &mut self.phase {
			Phase::WarpProof { .. } | Phase::TargetBlock(_) | Phase::PendingTargetBlock { .. } => {
				log::debug!(target: "sync", "Unexpected state response");
				ImportResult::BadResponse
			},
			Phase::State(sync) => sync.import(response),
		}
	}

	/// Get candidate for warp/block request.
	fn random_synced_available_peer(&self) -> Option<(&PeerId, &Peer<B>)> {
		let mut targets: Vec<_> = self.peers.values().map(|p| p.best_number).collect();
		if !targets.is_empty() {
			targets.sort();
			let median = targets[targets.len() / 2];
			// Find a random peer that is synced as much as peer majority.
			for (peer_id, peer) in self.peers.iter_mut() {
				if peer.state.is_available() && peer.best_number >= median {
					return Some((peer_id, peer))
				}
			}
		}

		None
	}

	/// Produce next warp proof request.
	fn next_warp_proof_request(&self) -> Option<(PeerId, WarpProofRequest<B>)> {
		let Phase::WarpProof { last_hash, .. } = &self.phase else {
			return None
		};

		if self.peers.iter().any(|(_, peer)| matches!(peer.state, PeerState::DownloadingProofs)) {
			// Only one warp proof request at a time is possible.
			return None
		}

		let Some((peer_id, peer)) = self.random_synced_available_peer() else {
			return None
		};

		trace!(target: LOG_TARGET, "New WarpProofRequest to {peer_id}, begin hash: {last_hash}.");
		peer.state = PeerState::DownloadingProofs;

		Some((*peer_id, WarpProofRequest { begin: *last_hash }))
	}

	/// Produce next target block request.
	pub fn next_target_block_request(&self) -> Option<(NumberFor<B>, BlockRequest<B>)> {
		match &self.phase {
			Phase::WarpProof { .. } | Phase::State(_) | Phase::PendingTargetBlock { .. } => None,
			Phase::TargetBlock(header) => {
				let request = BlockRequest::<B> {
					id: 0,
					fields: BlockAttributes::HEADER |
						BlockAttributes::BODY | BlockAttributes::JUSTIFICATION,
					from: FromBlock::Hash(header.hash()),
					direction: Direction::Ascending,
					max: Some(1),
				};
				Some((*header.number(), request))
			},
		}
	}

	/// Produce next state request.
	pub fn next_state_request(&self) -> Option<StateRequest> {
		match &self.phase {
			Phase::WarpProof { .. } | Phase::TargetBlock(_) | Phase::PendingTargetBlock { .. } =>
				None,
			Phase::State(sync) => Some(sync.next_request()),
		}
	}

	/// Return target block hash if it is known.
	pub fn target_block_hash(&self) -> Option<B::Hash> {
		match &self.phase {
			Phase::WarpProof { .. } | Phase::TargetBlock(_) | Phase::PendingTargetBlock { .. } =>
				None,
			Phase::State(s) => Some(s.target()),
		}
	}

	/// Return target block number if it is known.
	pub fn target_block_number(&self) -> Option<NumberFor<B>> {
		match &self.phase {
			Phase::WarpProof { .. } | Phase::PendingTargetBlock { .. } => None,
			Phase::TargetBlock(header) => Some(*header.number()),
			Phase::State(s) => Some(s.target_block_num()),
		}
	}

	/// Check if the state is complete.
	pub fn is_complete(&self) -> bool {
		match &self.phase {
			Phase::WarpProof { .. } | Phase::TargetBlock(_) | Phase::PendingTargetBlock { .. } =>
				false,
			Phase::State(sync) => sync.is_complete(),
		}
	}

	/// Returns state sync estimated progress (percentage, bytes)
	pub fn progress(&self) -> WarpSyncProgress<B> {
		match &self.phase {
			Phase::WarpProof { .. } => WarpSyncProgress {
				phase: WarpSyncPhase::DownloadingWarpProofs,
				total_bytes: self.total_proof_bytes,
			},
			Phase::TargetBlock(_) => WarpSyncProgress {
				phase: WarpSyncPhase::DownloadingTargetBlock,
				total_bytes: self.total_proof_bytes,
			},
			Phase::PendingTargetBlock { .. } => WarpSyncProgress {
				phase: WarpSyncPhase::AwaitingTargetBlock,
				total_bytes: self.total_proof_bytes,
			},
			Phase::State(sync) => WarpSyncProgress {
				phase: if self.is_complete() {
					WarpSyncPhase::ImportingState
				} else {
					WarpSyncPhase::DownloadingState
				},
				total_bytes: self.total_proof_bytes + sync.progress().size,
			},
		}
	}

	/// Get actions that should be performed by the owner on [`WarpSync`]'s behalf
	#[must_use]
	pub fn actions(&mut self) -> impl Iterator<Item = WarpSyncAction<B>> {

	}
}
