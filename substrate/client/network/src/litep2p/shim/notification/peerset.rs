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

//! `Peerset` implementation for `litep2p`.
//!
//! `Peerset` is a separate but related component running alongside the notification protocol,
//! responsible for maintaining connectivity to remote peers. `Peerset` has an imperfect view of the
//! network as the notification protocol is behind an asynchronous channel. Based on this imperfect
//! view, it tries to connect to remote peers and disconnect peers that should be disconnected from.
//!
//! It is these state transitions that are

//! `Peerset` deals with two types of peers:
//!  - regular pers
//!  - reserved peers
//!
//! Each protocol

use crate::{
	litep2p::peerstore::PeerstoreHandle,
	service::traits::{Direction, ValidationResult},
	ProtocolName,
};

use futures::{future::BoxFuture, stream::FuturesUnordered, Stream, StreamExt};
use futures_timer::Delay;

use litep2p::protocol::notification::NotificationError;
use sc_network_types::PeerId;
use sc_utils::mpsc::{tracing_unbounded, TracingUnboundedReceiver, TracingUnboundedSender};

use std::{
	collections::{hash_map::Entry, HashMap, HashSet},
	pin::Pin,
	task::{Context, Poll},
	time::Duration,
};

// TODO: should reserved set updates be atomic?

/// Logging target for the file.
const LOG_TARGET: &str = "sub-libp2p::peerset";

/// Default backoff for connection re-attempts.
const DEFAULT_BACKOFF: Duration = Duration::from_secs(30);

/// Reputation adjustment when a peer gets disconnected.
///
/// Lessens the likelyhood of the peer getting selected for an outbound connection soon.
const DISCONNECT_ADJUSTMENT: i32 = -256;

/// Reputation adjustment when a substream fails to open.
///
/// Lessens the likelyhood of the peer getting selected for an outbound connection soon.
const OPEN_FAILURE_ADJUSTMENT: i32 = -256;

/// Commands emitted by other subsystems of the blockchain to [`Peerset`].
#[derive(Debug)]
pub enum PeersetCommand {
	/// Set current reserved peer set.
	///
	/// This command removes all reserved peers that are not in `peers`.
	SetReservedPeers {
		/// New seserved peer set.
		peers: HashSet<PeerId>,
	},

	/// Add one or more reserved peers.
	///
	/// This command doesn't remove any reserved peers but only add new peers.
	AddReservedPeers {
		/// Reserved peers to add.
		peers: HashSet<PeerId>,
	},

	/// Remove reserved peers.
	RemoveReservedPeers {
		/// Reserved peers to remove.
		peers: HashSet<PeerId>,
	},

	/// Set reserved-only mode to true/false.
	SetReservedOnly {
		/// Should the protocol only accept/establish connections to reserved peers.
		reserved_only: bool,
	},

	/// Disconnect peer.
	DisconnectPeer {
		/// Peer ID.
		peer: PeerId,
	},
}

/// Commands emitted by [`Peerset`] to the notification protocol.
#[derive(Debug)]
pub enum PeersetNotificationCommand {
	/// Open substreams to one or more peers.
	OpenSubstream {
		/// Peer IDs.
		peers: Vec<PeerId>,
	},

	/// Close substream to one or more peers.
	CloseSubstream {
		/// Peer IDs.
		peers: Vec<PeerId>,
	},
}

/// Peer state.
///
/// Peer can be in 4 different state:
///  - disconnected
///  - connected
///  - connection is opening
///  - connection is closing
///  - connection is backed-off
///
/// Opening and closing are separate states as litep2p guarantees to report when the substream is
/// either fully open or fully closed and the slot allocation for opening a substream is tied to a
/// state transition which moves the peer to [`PeerState::Opening`]. This is because it allows
/// reserving a slot for peer to prevent infinite outbound substreams. If the substream is opened
/// successfully, peer is moved to state [`PeerState::Open`] but there is no modificatoin to the
/// slot count as an outbound slot was already allocated for the peer. If the substream fails to
/// open, the event is reported by litep2p and [`Peerset::report_substream_open_failure()`] is
/// called which will decrease the outbound slot count. Similarly for inbound streams, the slot is
/// allocated in [`Peerset::report_inbound_substream()`] which will prevent `Peerset` from accepting
/// infinite inbound substreams. If the inbound substream fails to open and since [`Peerset`] was
/// notified of it, litep2p will report the open failure and the inbound slot count is once again
/// decreased in [`Peerset::report_substream_open_failure()`]. If the substream is opened
/// successfully, the slot count is not modified.
///
/// Since closing a substream is not instantaneous, there is a separate [`PeersState::Closing`]
/// state which indicates that the substream is being closed but hasn't been closed by litep2p yet.
/// This state is used to prevent invalid state transitions where, for example, [`Peerset`] would
/// close a substream and then try to reopen it immediately.
///
/// Irrespective of which side closed the substream (local/remote), the substream is chilled for a
/// small amount of time ([`DEFAULT_BACKOFF`]) and during this time no inbound or outbound
/// substreams are accepted/established. Inbound subtreams are completely rejected and outbound
/// substream requests are put on hold. Once the backoff expires and if the local node is still
/// interested in opening an outbound substream (the request hasn't been canceled), outbound
/// substream request is made to litep2p.
///
/// Disconnections and open failures will contribute negatively to the peer score to prevent it from
/// being selected for another outbound substream request soon after the failure/disconnection. The
/// reputation decays towards zero over time and eventually the peer will be as likely to be
/// selected for an outbound substream as any other freshly added peer.
// TODO(aaro): actually implement what is specified here
// TODO(aaro): add lots of tests for different state transitions.
#[derive(Debug)]
enum PeerState {
	/// No active connection to peer.
	Disconnected,

	/// Connection to peer is pending.
	Opening {
		/// Direction of the connection.
		direction: Direction,
	},

	/// Substream to peer was recently closed and the peer is currently backed off.
	///
	/// Backoff only applies to outbound substreams. Inbound substream will not experience any sort
	/// of "banning" even if the peer is backed off and an inbound substream for the peer is
	/// received.
	Backoff,

	// Connected to peer.
	Connected {
		/// Is the peer inbound or outbound.
		direction: Direction,
	},

	// TODO:
	Canceled {
		/// Is the peer inbound or outbound.
		direction: Direction,
	},

	/// Connection to peer is closing.
	///
	/// State implies that the substream was asked to be closed by the local node and litep2p is
	/// closing the substream. No command modifying the connection state is accepted until the
	/// state has been set to [`PeerState::Disconnected`].
	Closing {
		/// Is the peer inbound or outbound.
		direction: Direction,

		/// Was the peer reserved peer when it was disconnected.
		was_reserved: bool,
	},
}

/// Peer context.
#[derive(Debug)]
struct PeerContext {
	/// Is the peer a reserved peer.
	is_reserved: bool,

	/// Peer state.
	state: PeerState,
}

/// `Peerset` implementation.
///
/// `Peerset` allows other subsystems of the blockchain to modify the connection state
/// of the notification protocol by adding and removing reserved peers.
///
/// `Peerset` is also responsible for maintaining the desired amount of peers the protocol is
/// connected to by establishing outbound connections and accepting/rejecting inbound connections.
#[derive(Debug)]
pub struct Peerset {
	/// Protocol name.
	protocol: ProtocolName,

	/// RX channel for receiving commands.
	cmd_rx: TracingUnboundedReceiver<PeersetCommand>,

	/// Maximum number of outbound peers.
	max_out: usize,

	/// Current number of outbound peers.
	num_out: usize,

	/// Maximum number of inbound peers.
	max_in: usize,

	/// Current number of inbound peers.
	num_in: usize,

	/// Only connect to/accept connections from reserved peers.
	reserved_only: bool,

	/// Current reserved peer set.
	reserved_peers: HashSet<PeerId>,

	/// Handle to `Peerstore`.
	peerstore_handle: PeerstoreHandle,

	/// Peers.
	peers: HashMap<PeerId, PeerState>,

	/// Pending cancels.
	pending_cancels: HashSet<PeerId>,

	/// Pending backoffs for peers who recently disconnected.
	pending_backoffs: FuturesUnordered<BoxFuture<'static, PeerId>>,
}

impl Peerset {
	/// Create new [`Peerset`].
	pub fn new(
		protocol: ProtocolName,
		max_out: usize,
		max_in: usize,
		reserved_only: bool,
		reserved_peers: HashSet<PeerId>,
		mut peerstore_handle: PeerstoreHandle,
	) -> (Self, TracingUnboundedSender<PeersetCommand>) {
		let (cmd_tx, cmd_rx) = tracing_unbounded("mpsc-peerset-protocol", 100_000);
		let peers = reserved_peers
			.iter()
			.map(|peer| (*peer, PeerState::Disconnected))
			.collect::<HashMap<_, _>>();

		// register protocol's commad channel to `Peerstore` so it can issue disconnect commands
		// if some connected peer gets banned.
		peerstore_handle.register_protocol(cmd_tx.clone());

		(
			Self {
				protocol,
				max_out,
				num_out: 0usize,
				max_in,
				num_in: 0usize,
				reserved_peers,
				cmd_rx,
				peerstore_handle,
				reserved_only,
				peers,
				pending_backoffs: FuturesUnordered::new(),
				pending_cancels: HashSet::new(),
			},
			cmd_tx,
		)
	}

	/// Report to [`Peerset`] that a substream was opened.
	///
	/// Slot for the stream was "preallocated" when the it was initiated (outbound) or accepted
	/// (inbound) by the local node which is why this function doesn't allocate a slot for the peer.
	pub fn report_substream_opened(&mut self, peer: PeerId, direction: Direction) {
		log::trace!(
			target: LOG_TARGET,
			"{}: substream opened to {peer:?}, direction {direction:?}, reserved peer {}",
			self.protocol,
			self.reserved_peers.contains(&peer)
		);

		self.peers.insert(peer, PeerState::Connected { direction });
	}

	/// Report to [`Peerset`] that a substream was closed.
	///
	/// If the peer was not a reserved peer, the inbound/outbound slot count is adjusted to account
	/// for the disconnected peer. After the connection is closed, the peer is chilled for a
	/// duration of [`DEFAULT_BACKOFF`] which prevens [`Peerset`] from establishing/accepting new
	/// connections for that time period.
	///
	/// Reserved peers cannot be disconnected using this method and they can be disconnected only if
	/// they're banned.
	pub fn report_substream_closed(&mut self, peer: PeerId) {
		log::trace!(
			target: LOG_TARGET,
			"{}: substream closed to {peer:?}, reserved peer {}",
			self.protocol,
			self.reserved_peers.contains(&peer)
		);

		let Some(state) = self.peers.get_mut(&peer) else {
			log::debug!(target: LOG_TARGET, "{}: substream closed for unknown peer {peer:?}", self.protocol);
			return
		};

		match (self.reserved_peers.contains(&peer), &state) {
			(false, PeerState::Connected { direction: Direction::Inbound }) => {
				self.num_in -= 1;
			},
			(_, PeerState::Closing { direction: Direction::Inbound, was_reserved }) =>
				if !was_reserved {
					self.num_in -= 1;
				},
			(false, PeerState::Connected { direction: Direction::Outbound }) => {
				self.num_out -= 1;
			},
			(_, PeerState::Closing { direction: Direction::Outbound, was_reserved }) =>
				if !was_reserved {
					self.num_out -= 1;
				},
			(true, PeerState::Closing { .. } | PeerState::Connected { .. }) => {
				log::debug!(target: LOG_TARGET, "{}: reserved peer {peer:?} disconnected", self.protocol);
			},
			(_, state) => {
				log::warn!(target: LOG_TARGET, "{}: invalid state for disconnected peer {peer:?}: {state:?}", self.protocol);
				debug_assert!(false);
			},
		}
		*state = PeerState::Backoff;

		self.peerstore_handle.report_peer(peer, DISCONNECT_ADJUSTMENT);
		self.pending_backoffs.push(Box::pin(async move {
			Delay::new(DEFAULT_BACKOFF).await;
			peer
		}));
	}

	/// Report to [`Peerset`] that an inbound substream was opened and that it should validate it.
	pub fn report_inbound_substream(&mut self, peer: PeerId) -> ValidationResult {
		log::trace!(target: LOG_TARGET, "{}: inbound substream from {peer:?}", self.protocol);

		let state = self.peers.entry(peer).or_insert(PeerState::Disconnected);

		match state {
			PeerState::Disconnected => {
				*state = PeerState::Opening { direction: Direction::Inbound };
			},
			PeerState::Backoff => {
				log::trace!(target: LOG_TARGET, "{}: ({peer:?}) is backed-off, reject inbound substream", self.protocol);
				return ValidationResult::Reject
			},
			PeerState::Opening { direction: Direction::Outbound } => {
				log::trace!(
					target: LOG_TARGET,
					"{}: inbound substream received for {peer:?} that was marked outbound",
					self.protocol
				);

				if !self.reserved_peers.contains(&peer) {
					if self.num_in >= self.num_out {
						log::debug!(
							target: LOG_TARGET,
							"{}: inbound substream for {peer:?} cannot be accepted because there aren't free slots",
							self.protocol
						);

						self.num_out -= 1;
						*state = PeerState::Disconnected;
						return ValidationResult::Reject
					}

					self.num_in += 1;
				}

				*state = PeerState::Opening { direction: Direction::Inbound };
				return ValidationResult::Accept
			},
			PeerState::Canceled { direction } => {
				todo!("handle inbound substream for canceled {peer:?}");
			},
			state => {
				log::warn!(target: LOG_TARGET, "{}: invalid state ({state:?}) for inbound substream, peer {peer:?}", self.protocol);
				debug_assert!(false);
				return ValidationResult::Reject
			},
		}

		if self.reserved_peers.contains(&peer) {
			return ValidationResult::Accept
		}

		if self.num_in < self.max_in {
			self.num_in += 1;
			return ValidationResult::Accept
		}

		log::trace!(target: LOG_TARGET, "{}: reject {peer:?}, not a reserved peer and no free inbound slots", self.protocol);

		*state = PeerState::Disconnected;
		return ValidationResult::Reject
	}

	/// Report to [`Peerset`] that an inbound substream was opened and that it should validate it.
	pub fn report_substream_open_failure(&mut self, peer: PeerId, error: NotificationError) {
		log::trace!(target: LOG_TARGET, "{}: failed to open substream to {peer:?}: {error:?}", self.protocol);

		match self.peers.get(&peer) {
			Some(PeerState::Opening { direction }) =>
				if !self.reserved_peers.contains(&peer) {
					match direction {
						Direction::Inbound => match self.num_in.checked_sub(1) {
							Some(value) => {
								self.num_in = value;
							},
							None => {
								panic!("tried to substract from zero, inbound {peer:?}");
							},
						},
						Direction::Outbound => match self.num_out.checked_sub(1) {
							Some(value) => {
								self.num_out = value;
							},
							None => {
								panic!("tried to substract from zero, outbound {peer:?}");
							},
						},
					}
				},
			Some(PeerState::Canceled { direction }) => {
				log::debug!(target: LOG_TARGET, "substream open failure for canceled peer {peer:?}, {direction:?}");
			},
			state => {
				log::warn!(target: LOG_TARGET, "unexpected state for substream open failure: {peer:?} {state:?}");
				debug_assert!(false);
			},
		}

		self.peers.insert(peer, PeerState::Backoff);
		self.peerstore_handle.report_peer(peer, OPEN_FAILURE_ADJUSTMENT);
		self.pending_backoffs.push(Box::pin(async move {
			Delay::new(DEFAULT_BACKOFF).await;
			peer
		}));
	}
}

impl Stream for Peerset {
	type Item = PeersetNotificationCommand;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		// check if any pending backoffs have expired
		while let Poll::Ready(Some(peer)) = self.pending_backoffs.poll_next_unpin(cx) {
			log::trace!(target: LOG_TARGET, "{}: backoff expired for {peer:?}", self.protocol);
			self.peers.insert(peer, PeerState::Disconnected);
		}

		if let Poll::Ready(Some(action)) = Pin::new(&mut self.cmd_rx).poll_next(cx) {
			match action {
				PeersetCommand::DisconnectPeer { peer } => match self.peers.remove(&peer) {
					Some(PeerState::Connected { direction }) => {
						log::trace!(target: LOG_TARGET, "close connection to {peer:?}, direction {direction:?}");

						self.peers
							.insert(peer, PeerState::Closing { direction, was_reserved: false });
						return Poll::Ready(Some(PeersetNotificationCommand::CloseSubstream {
							peers: vec![peer],
						}))
					},
					Some(PeerState::Backoff) => {
						log::trace!(target: LOG_TARGET, "{}: cannot disconnect {peer:?}, already backed-off", self.protocol);
						self.peers.insert(peer, PeerState::Backoff);
					},
					Some(PeerState::Opening { .. }) => {
						todo!("queue pending close for the stream and once it opens, close the stream");
					},
					Some(state) => {
						log::warn!(target: LOG_TARGET, "{}: cannot disconnect {peer:?}, invalid state: {state:?}", self.protocol);
						self.peers.insert(peer, state);
						debug_assert!(false);
					},
					None => {
						log::error!(target: LOG_TARGET, "{}: {peer:?} doens't exist", self.protocol);
						debug_assert!(false);
					},
				},
				PeersetCommand::SetReservedPeers { peers } => {
					log::warn!(target: LOG_TARGET, "{}: set reserved peers {peers:?}", self.protocol);

					if !peers.is_empty() {
						panic!("set reserved peers");
					}

					// self.reserved_peers = peers;
				},
				PeersetCommand::AddReservedPeers { peers } => {
					log::debug!(target: LOG_TARGET, "{}: add reserved peers {peers:?}", self.protocol);

					return Poll::Ready(Some(PeersetNotificationCommand::OpenSubstream {
						peers: peers
							.iter()
							.filter_map(|peer| {
								if !self.reserved_peers.insert(*peer) {
									log::warn!(
										target: LOG_TARGET,
										"{}: {peer:?} is already a reserved peer",
										self.protocol,
									);
									return None
								}

								std::matches!(
									self.peers.get_mut(peer),
									None | Some(PeerState::Disconnected)
								)
								.then(|| {
									self.peers.insert(
										*peer,
										PeerState::Opening { direction: Direction::Outbound },
									);
									*peer
								})
							})
							.collect(),
					}))
				},
				PeersetCommand::RemoveReservedPeers { peers } => {
					log::debug!(target: LOG_TARGET, "{}: remove reserved peers {peers:?}", self.protocol);

					return Poll::Ready(Some(PeersetNotificationCommand::CloseSubstream {
						peers: peers
							.iter()
							.filter_map(|peer| {
								if !self.reserved_peers.remove(peer) {
									log::warn!(
										target: LOG_TARGET,
										"{}: {peer} is not a reserved peer",
										self.protocol,
									);
									return None
								}

								let peer_state = self.peers.get_mut(peer)?;

								match std::mem::replace(peer_state, PeerState::Disconnected) {
									PeerState::Connected { direction } => {
										*peer_state =
											PeerState::Closing { direction, was_reserved: true };
										return Some(*peer)
									},
									PeerState::Opening { direction } => {
										*peer_state = PeerState::Canceled { direction };
										self.pending_cancels.insert(*peer);
										None
									},
									PeerState::Backoff => {
										*peer_state = PeerState::Backoff;
										None
									},
									state => {
										log::warn!(target: LOG_TARGET, "ignore state {state:?} for {peer:?}");
										*peer_state = state;
										debug_assert!(false);
										None
									},
								}
							})
							.collect(),
					}))
				},
				PeersetCommand::SetReservedOnly { reserved_only } => {
					log::debug!(target: LOG_TARGET, "{}: set reserved only mode to {reserved_only}", self.protocol);
					panic!("set reserved only");

					// self.reserved_only = reserved_only;
					// if self.reserved_only {}
					// TODO(aaro): this should return multiple peers perhaps?
				},
			}
		}

		// try to establish connection any reserved peer who is not currently connected
		let reserved_peers = self
			.peers
			.iter()
			.filter_map(|(peer, state)| {
				(self.reserved_peers.contains(peer) &&
					std::matches!(state, PeerState::Disconnected))
				.then_some(*peer)
			})
			.collect::<Vec<_>>();

		if !reserved_peers.is_empty() {
			log::trace!(
				target: LOG_TARGET,
				"{}: start connecting to reserved peers {reserved_peers:?}",
				self.protocol,
			);

			reserved_peers.iter().for_each(|peer| {
				self.peers.insert(*peer, PeerState::Opening { direction: Direction::Outbound });
			});

			return Poll::Ready(Some(PeersetNotificationCommand::OpenSubstream {
				peers: reserved_peers,
			}))
		}

		// if the number of outbound peers is lower than the desired amount of oubound peers,
		// query `PeerStore` and try to get a new outbound candidated.
		if self.num_out < self.max_out && !self.reserved_only {
			// TODO(aaro): continuously update the ignore list so it doens't have to be recollected
			// every time
			let ignore: HashSet<&PeerId> = self
				.peers
				.iter()
				.filter_map(|(peer, state)| {
					std::matches!(
						state,
						PeerState::Closing { .. } |
							PeerState::Backoff | PeerState::Opening { .. } |
							PeerState::Connected { .. } | PeerState::Canceled { .. },
					)
					.then_some(peer)
				})
				.collect();

			let peers: Vec<_> = self
				.peerstore_handle
				.next_outbound_peers(&ignore, self.max_out - self.num_out)
				.collect();

			if peers.len() > 0 {
				log::trace!(target: LOG_TARGET, "{}: start connecting to peers {peers:?}", self.protocol);

				peers.iter().for_each(|peer| {
					self.peers.insert(*peer, PeerState::Opening { direction: Direction::Outbound });
				});
				self.num_out += peers.len();

				return Poll::Ready(Some(PeersetNotificationCommand::OpenSubstream { peers }))
			}
		}

		Poll::Pending
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::litep2p::peerstore::peerstore_handle;

	#[tokio::test]
	async fn test() {
		let mut peerstore_handle = peerstore_handle();
		let (mut peerset, mut tx) = Peerset::new(
			ProtocolName::from("/notif/1"),
			8usize,
			32usize,
			false,
			HashSet::new(),
			peerstore_handle.clone(),
		);

		peerset.report_substream_opened(PeerId::random(), Direction::Inbound);
	}

	#[tokio::test]
	async fn set_reserved_peers() {
		let reserved_peer1 = PeerId::random();
		let reserved_peer2 = PeerId::random();
		let reserved_peer3 = PeerId::random();
		let mut peerstore_handle = peerstore_handle();

		let (mut peerset, mut tx) = Peerset::new(
			ProtocolName::from("/notif/1"),
			8usize,
			32usize,
			false,
			HashSet::from_iter([reserved_peer1, reserved_peer2, reserved_peer3]),
			peerstore_handle.clone(),
		);

		match peerset.next().await {
			state => panic!("didn't expect {state:?}"),
			Some(PeersetNotificationCommand::OpenSubstream { peers }) => {
				todo!();
			},
		}
	}

	#[tokio::test]
	async fn set_reserved_only() {}
}
