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

//! `Peerstore` implementation for `litep2p`.
//!
//! `Peerstore` is responsible for storing information about remote peers
//! such as their addresses, reputations, supported protocols etc.

use crate::{
	litep2p::shim::notification::peerset::PeersetCommand, peer_store::PeerStoreProvider,
	protocol_controller::ProtocolHandle, service::traits::PeerStore, ObservedRole,
	ReputationChange,
};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use wasm_timer::Delay;

use sc_network_types::PeerId;
use sc_utils::mpsc::TracingUnboundedSender;

use std::{
	collections::{HashMap, HashSet},
	sync::Arc,
	time::{Duration, Instant},
};

/// Logging target for the file.
const LOG_TARGET: &str = "sub-libp2p::peerstore";

/// We don't accept nodes whose reputation is under this value.
pub const BANNED_THRESHOLD: i32 = 82 * (i32::MIN / 100);

/// Reputation change for a node when we get disconnected from it.
const DISCONNECT_REPUTATION_CHANGE: i32 = -256;

/// Relative decrement of a reputation value that is applied every second. I.e., for inverse
/// decrement of 50 we decrease absolute value of the reputation by 1/50. This corresponds to a
/// factor of `k = 0.98`. It takes ~ `ln(0.5) / ln(k)` seconds to reduce the reputation by half,
/// or 34.3 seconds for the values above. In this setup the maximum allowed absolute value of
/// `i32::MAX` becomes 0 in ~1100 seconds (actually less due to integer arithmetic).
const INVERSE_DECREMENT: i32 = 50;

/// Amount of time between the moment we last updated the [`PeerStore`] entry and the moment we
/// remove it, once the reputation value reaches 0.
const FORGET_AFTER: Duration = Duration::from_secs(3600);

/// Peer information.
#[derive(Debug, Clone, Copy)]
struct PeerInfo {
	/// Reputation of the peer.
	reputation: i32,

	/// Instant when the peer was last updated.
	last_updated: Instant,

	/// Role of the peer, if known.
	role: Option<ObservedRole>,
}

impl PeerInfo {
	fn is_banned(&self) -> bool {
		self.reputation < BANNED_THRESHOLD
	}

	fn add_reputation(&mut self, increment: i32) {
		self.reputation = self.reputation.saturating_add(increment);
		self.bump_last_updated();
	}

	fn decay_reputation(&mut self, seconds_passed: u64) {
		// Note that decaying the reputation value happens "on its own",
		// so we don't do `bump_last_updated()`.
		for _ in 0..seconds_passed {
			let mut diff = self.reputation / INVERSE_DECREMENT;
			if diff == 0 && self.reputation < 0 {
				diff = -1;
			} else if diff == 0 && self.reputation > 0 {
				diff = 1;
			}

			self.reputation = self.reputation.saturating_sub(diff);

			if self.reputation == 0 {
				break
			}
		}
	}

	fn bump_last_updated(&mut self) {
		self.last_updated = Instant::now();
	}
}

#[derive(Debug, Default)]
pub struct PeerstoreHandleInner {
	peers: HashMap<PeerId, PeerInfo>,
	protocols: Vec<TracingUnboundedSender<PeersetCommand>>,
}

#[derive(Debug, Clone, Default)]
pub struct PeerstoreHandle(Arc<Mutex<PeerstoreHandleInner>>);

impl PeerstoreHandle {
	/// Register protocol to `PeerstoreHandle`.
	///
	/// This channel is only used to disconnect banned peers and may be replaced
	/// with something else in the future.
	pub fn register_protocol(&mut self, sender: TracingUnboundedSender<PeersetCommand>) {
		self.0.lock().protocols.push(sender);
	}

	/// Add known peer to [`Peerstore`].
	pub fn add_known_peer(&mut self, peer: PeerId) {
		self.0
			.lock()
			.peers
			.insert(peer, PeerInfo { reputation: 0i32, last_updated: Instant::now(), role: None });
	}

	/// Adjust peer reputation.
	pub fn report_peer(&mut self, peer: PeerId, reputation_change: i32) {
		let mut lock = self.0.lock();

		match lock.peers.get_mut(&peer) {
			Some(info) => {
				info.reputation = info.reputation.saturating_add(reputation_change);
			},
			None => {
				lock.peers.insert(
					peer,
					PeerInfo {
						reputation: reputation_change,
						last_updated: Instant::now(),
						role: None,
					},
				);
			},
		}

		if lock
			.peers
			.get(&peer)
			.expect("peer exist since it was just modified; qed")
			.is_banned()
		{
			log::debug!(target: LOG_TARGET, "{peer:?} banned, disconnecting");

			for sender in &lock.protocols {
				sender.unbounded_send(PeersetCommand::DisconnectPeer { peer });
			}
		}
	}

	/// Get next outbound peers for connection attempts, ignoring all peers in `ignore`.
	///
	/// Returns `None` if there are no peers available.
	pub fn next_outbound_peers(
		&self,
		ignore: &HashSet<&PeerId>,
		num_peers: usize,
	) -> impl Iterator<Item = PeerId> {
		let handle = self.0.lock();

		// TODO: this is really bad
		let mut candidates = handle
			.peers
			.iter()
			.filter_map(|(peer, info)| {
				(!ignore.contains(&peer) && !info.is_banned()).then_some((*peer, info.reputation))
			})
			.collect::<Vec<(PeerId, _)>>();
		candidates.sort_by(|(_, a), (_, b)| a.cmp(b));
		candidates
			.into_iter()
			.take(num_peers)
			.map(|(peer, _score)| peer)
			.collect::<Vec<_>>()
			.into_iter()
	}

	pub fn peer_count(&self) -> usize {
		self.0.lock().peers.len()
	}

	fn progress_time(&self, seconds_passed: u64) {
		if seconds_passed == 0 {
			return
		}

		let mut lock = self.0.lock();

		// Drive reputation values towards 0.
		lock.peers
			.iter_mut()
			.for_each(|(_, info)| info.decay_reputation(seconds_passed));

		// Retain only entries with non-zero reputation values or not expired ones.
		let now = Instant::now();
		lock.peers
			.retain(|_, info| info.reputation != 0 || info.last_updated + FORGET_AFTER > now);
	}
}

impl PeerStoreProvider for PeerstoreHandle {
	fn is_banned(&self, peer_id: &PeerId) -> bool {
		todo!();
	}

	/// Register a protocol handle to disconnect peers whose reputation drops below the threshold.
	fn register_protocol(&self, protocol_handle: ProtocolHandle) {
		todo!();
	}

	/// Report peer disconnection for reputation adjustment.
	fn report_disconnect(&self, peer_id: PeerId) {
		todo!();
	}

	/// Adjust peer reputation.
	fn report_peer(&self, peer_id: PeerId, change: ReputationChange) {
		todo!();
	}

	/// Set peer role.
	fn set_peer_role(&self, peer_id: &PeerId, role: ObservedRole) {
		todo!();
	}

	/// Get peer reputation.
	fn peer_reputation(&self, peer_id: &PeerId) -> i32 {
		todo!();
	}

	/// Get peer role, if available.
	fn peer_role(&self, peer_id: &PeerId) -> Option<ObservedRole> {
		todo!();
	}

	/// Get candidates with highest reputations for initiating outgoing connections.
	fn outgoing_candidates(&self, count: usize, ignored: HashSet<PeerId>) -> Vec<PeerId> {
		todo!();
	}

	/// Get the number of known peers.
	///
	/// This number might not include some connected peers in rare cases when their reputation
	/// was not updated for one hour, because their entries in [`PeerStore`] were dropped.
	fn num_known_peers(&self) -> usize {
		self.0.lock().peers.len()
	}

	/// Add known peer.
	fn add_known_peer(&self, peer_id: PeerId) {
		todo!();
	}
}

// TODO: documentation
// TODO: make acquiring this part of the interface and instantiate the correct `Peerstore`
// when starting the backend
static PEERSET_HANDLE: Lazy<PeerstoreHandle> =
	Lazy::new(|| PeerstoreHandle(Arc::new(Mutex::new(Default::default()))));

/// Get handle to `Peerstore`.
pub fn peerstore_handle() -> PeerstoreHandle {
	Lazy::force(&PEERSET_HANDLE).clone()
}

/// Peerstore implementation.
pub struct Peerstore {
	/// Handle to `Peerstore`.
	peerstore_handle: PeerstoreHandle,
}

impl Peerstore {
	/// Create new [`Peerstore`].
	pub fn new() -> Self {
		Self { peerstore_handle: peerstore_handle() }
	}

	/// Get mutable reference to the underlying [`PeerstoreHandle`].
	pub fn handle(&mut self) -> &mut PeerstoreHandle {
		&mut self.peerstore_handle
	}

	/// Add known peer to [`Peerstore`].
	pub fn add_known_peer(&mut self, peer: PeerId) {
		self.peerstore_handle.add_known_peer(peer);
	}

	/// Start [`Peerstore`] event loop.
	async fn run(self) {
		let started = Instant::now();
		let mut latest_time_update = started;

		loop {
			let now = Instant::now();
			// We basically do `(now - self.latest_update).as_secs()`, except that by the way we do
			// it we know that we're not going to miss seconds because of rounding to integers.
			let seconds_passed = {
				let elapsed_latest = latest_time_update - started;
				let elapsed_now = now - started;
				latest_time_update = now;
				elapsed_now.as_secs() - elapsed_latest.as_secs()
			};

			self.peerstore_handle.progress_time(seconds_passed);
			let _ = Delay::new(Duration::from_secs(1)).await;
		}
	}
}

#[async_trait::async_trait]
impl PeerStore for Peerstore {
	/// Get handle to `PeerStore`.
	fn handle(&self) -> Arc<dyn PeerStoreProvider> {
		Arc::new(peerstore_handle())
	}

	/// Start running `PeerStore` event loop.
	async fn run(self) {
		self.run().await;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use sc_utils::mpsc::tracing_unbounded;

	#[test]
	fn acquire_mutual_handle() {
		// acquire first handle to peer store and register protocol
		let mut handle1 = peerstore_handle();
		let (tx1, _) = tracing_unbounded("mpsc-peerset-protocol", 100_000);
		handle1.register_protocol(tx1);

		// acquire second handle to peerstore and verify both handles have the registered protocol
		let mut handle2 = peerstore_handle();
		assert_eq!(handle1.0.lock().protocols.len(), 1);
		assert_eq!(handle2.0.lock().protocols.len(), 1);

		// register another protocol using the second handle and verify both handles have the
		// protocol
		let (tx2, _) = tracing_unbounded("mpsc-peerset-protocol", 100_000);
		handle1.register_protocol(tx2);
		assert_eq!(handle1.0.lock().protocols.len(), 2);
		assert_eq!(handle2.0.lock().protocols.len(), 2);
	}
}
