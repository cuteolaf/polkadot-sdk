// Copyright (C) Parity Technologies (UK) Ltd.
// This file is part of Cumulus.

// Cumulus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Cumulus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Cumulus.  If not, see <http://www.gnu.org/licenses/>.

//! Autogenerated weights for `pallet_assets`
//!
//! THIS FILE WAS AUTO-GENERATED USING THE SUBSTRATE BENCHMARK CLI VERSION 4.0.0-dev
//! DATE: 2023-06-20, STEPS: `50`, REPEAT: `20`, LOW RANGE: `[]`, HIGH RANGE: `[]`
//! WORST CASE MAP SIZE: `1000000`
//! HOSTNAME: `runner-ynta1nyy-project-238-concurrent-0`, CPU: `Intel(R) Xeon(R) CPU @ 2.60GHz`
//! EXECUTION: ``, WASM-EXECUTION: `Compiled`, CHAIN: `Some("asset-hub-kusama-dev")`, DB CACHE: 1024

// Executed Command:
// ./target/production/polkadot-parachain
// benchmark
// pallet
// --steps=50
// --repeat=20
// --extrinsic=*
// --execution=wasm
// --wasm-execution=compiled
// --heap-pages=4096
// --json-file=/var/lib/gitlab-runner/builds/zyw4fam_/0/parity/mirrors/cumulus/.git/.artifacts/bench.json
// --pallet=pallet_assets
// --no-storage-info
// --no-median-slopes
// --no-min-squares
// --extrinsic=*
// --steps=50
// --repeat=20
// --json
// --header=./file_header.txt
// --output=./parachains/runtimes/assets/asset-hub-kusama/src/weights/

#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(unused_parens)]
#![allow(unused_imports)]
#![allow(missing_docs)]

use frame_support::{traits::Get, weights::Weight};
use core::marker::PhantomData;

/// Weight functions for `pallet_assets`.
pub struct WeightInfo<T>(PhantomData<T>);
impl<T: frame_system::Config> pallet_assets::WeightInfo for WeightInfo<T> {
	/// Storage: `ParachainInfo::ParachainId` (r:1 w:0)
	/// Proof: `ParachainInfo::ParachainId` (`max_values`: Some(1), `max_size`: Some(4), added: 499, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `System::Account` (r:1 w:1)
	/// Proof: `System::Account` (`max_values`: None, `max_size`: Some(128), added: 2603, mode: `MaxEncodedLen`)
	fn create() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `107`
		//  Estimated: `4273`
		// Minimum execution time: 30_485_000 picoseconds.
		Weight::from_parts(31_007_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(3))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn force_create() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `4`
		//  Estimated: `4273`
		// Minimum execution time: 12_991_000 picoseconds.
		Weight::from_parts(13_304_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(1))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn start_destroy() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `276`
		//  Estimated: `4273`
		// Minimum execution time: 15_689_000 picoseconds.
		Weight::from_parts(16_063_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(1))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: ForeignAssets Asset (r:1 w:1)
	/// Proof: ForeignAssets Asset (max_values: None, max_size: Some(808), added: 3283, mode: MaxEncodedLen)
	/// Storage: ForeignAssets Account (r:1001 w:1000)
	/// Proof: ForeignAssets Account (max_values: None, max_size: Some(732), added: 3207, mode: MaxEncodedLen)
	/// Storage: System Account (r:1000 w:1000)
	/// Proof: System Account (max_values: None, max_size: Some(128), added: 2603, mode: MaxEncodedLen)
	/// The range of component `c` is `[0, 1000]`.
	/// The range of component `c` is `[0, 1000]`.
	fn destroy_accounts(c: u32, ) -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `0 + c * (208 ±0)`
		//  Estimated: `4273 + c * (3207 ±0)`
		// Minimum execution time: 18_533_000 picoseconds.
		Weight::from_parts(18_791_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			// Standard Error: 5_059
			.saturating_add(Weight::from_parts(12_049_659, 0).saturating_mul(c.into()))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().reads((2_u64).saturating_mul(c.into())))
			.saturating_add(T::DbWeight::get().writes(1))
			.saturating_add(T::DbWeight::get().writes((2_u64).saturating_mul(c.into())))
			.saturating_add(Weight::from_parts(0, 3207).saturating_mul(c.into()))
	}
	/// Storage: ForeignAssets Asset (r:1 w:1)
	/// Proof: ForeignAssets Asset (max_values: None, max_size: Some(808), added: 3283, mode: MaxEncodedLen)
	/// Storage: ForeignAssets Approvals (r:1001 w:1000)
	/// Proof: ForeignAssets Approvals (max_values: None, max_size: Some(746), added: 3221, mode: MaxEncodedLen)
	/// The range of component `a` is `[0, 1000]`.
	/// The range of component `a` is `[0, 1000]`.
	fn destroy_approvals(a: u32, ) -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `413 + a * (86 ±0)`
		//  Estimated: `4273 + a * (3221 ±0)`
		// Minimum execution time: 20_028_000 picoseconds.
		Weight::from_parts(20_148_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			// Standard Error: 3_401
			.saturating_add(Weight::from_parts(13_897_319, 0).saturating_mul(a.into()))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().reads((1_u64).saturating_mul(a.into())))
			.saturating_add(T::DbWeight::get().writes(1))
			.saturating_add(T::DbWeight::get().writes((1_u64).saturating_mul(a.into())))
			.saturating_add(Weight::from_parts(0, 3221).saturating_mul(a.into()))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Metadata` (r:1 w:0)
	/// Proof: `ForeignAssets::Metadata` (`max_values`: None, `max_size`: Some(738), added: 3213, mode: `MaxEncodedLen`)
	fn finish_destroy() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 15_949_000 picoseconds.
		Weight::from_parts(16_241_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	fn mint() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 27_156_000 picoseconds.
		Weight::from_parts(28_182_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	fn burn() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `350`
		//  Estimated: `4273`
		// Minimum execution time: 33_503_000 picoseconds.
		Weight::from_parts(33_860_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:2 w:2)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `System::Account` (r:1 w:1)
	/// Proof: `System::Account` (`max_values`: None, `max_size`: Some(128), added: 2603, mode: `MaxEncodedLen`)
	fn transfer() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `350`
		//  Estimated: `7404`
		// Minimum execution time: 45_065_000 picoseconds.
		Weight::from_parts(45_856_000, 0)
			.saturating_add(Weight::from_parts(0, 7404))
			.saturating_add(T::DbWeight::get().reads(4))
			.saturating_add(T::DbWeight::get().writes(4))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:2 w:2)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `System::Account` (r:1 w:1)
	/// Proof: `System::Account` (`max_values`: None, `max_size`: Some(128), added: 2603, mode: `MaxEncodedLen`)
	fn transfer_keep_alive() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `350`
		//  Estimated: `7404`
		// Minimum execution time: 39_913_000 picoseconds.
		Weight::from_parts(40_791_000, 0)
			.saturating_add(Weight::from_parts(0, 7404))
			.saturating_add(T::DbWeight::get().reads(4))
			.saturating_add(T::DbWeight::get().writes(4))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:2 w:2)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `System::Account` (r:1 w:1)
	/// Proof: `System::Account` (`max_values`: None, `max_size`: Some(128), added: 2603, mode: `MaxEncodedLen`)
	fn force_transfer() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `350`
		//  Estimated: `7404`
		// Minimum execution time: 45_337_000 picoseconds.
		Weight::from_parts(45_980_000, 0)
			.saturating_add(Weight::from_parts(0, 7404))
			.saturating_add(T::DbWeight::get().reads(4))
			.saturating_add(T::DbWeight::get().writes(4))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:0)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	fn freeze() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `350`
		//  Estimated: `4273`
		// Minimum execution time: 19_012_000 picoseconds.
		Weight::from_parts(19_326_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:0)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	fn thaw() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `350`
		//  Estimated: `4273`
		// Minimum execution time: 18_656_000 picoseconds.
		Weight::from_parts(19_205_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn freeze_asset() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `276`
		//  Estimated: `4273`
		// Minimum execution time: 15_440_000 picoseconds.
		Weight::from_parts(15_825_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(1))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn thaw_asset() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `276`
		//  Estimated: `4273`
		// Minimum execution time: 15_465_000 picoseconds.
		Weight::from_parts(15_769_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(1))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Metadata` (r:1 w:0)
	/// Proof: `ForeignAssets::Metadata` (`max_values`: None, `max_size`: Some(738), added: 3213, mode: `MaxEncodedLen`)
	fn transfer_ownership() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 16_579_000 picoseconds.
		Weight::from_parts(16_931_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn set_team() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 15_138_000 picoseconds.
		Weight::from_parts(15_435_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(1))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: ForeignAssets Asset (r:1 w:0)
	/// Proof: ForeignAssets Asset (max_values: None, max_size: Some(808), added: 3283, mode: MaxEncodedLen)
	/// Storage: ForeignAssets Metadata (r:1 w:1)
	/// Proof: ForeignAssets Metadata (max_values: None, max_size: Some(738), added: 3213, mode: MaxEncodedLen)
	/// The range of component `n` is `[0, 50]`.
	/// The range of component `s` is `[0, 50]`.
	/// The range of component `n` is `[0, 50]`.
	/// The range of component `s` is `[0, 50]`.
	fn set_metadata(_n: u32, _s: u32, ) -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 29_846_000 picoseconds.
		Weight::from_parts(31_607_649, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:0)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Metadata` (r:1 w:1)
	/// Proof: `ForeignAssets::Metadata` (`max_values`: None, `max_size`: Some(738), added: 3213, mode: `MaxEncodedLen`)
	fn clear_metadata() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `406`
		//  Estimated: `4273`
		// Minimum execution time: 30_582_000 picoseconds.
		Weight::from_parts(31_008_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: ForeignAssets Asset (r:1 w:0)
	/// Proof: ForeignAssets Asset (max_values: None, max_size: Some(808), added: 3283, mode: MaxEncodedLen)
	/// Storage: ForeignAssets Metadata (r:1 w:1)
	/// Proof: ForeignAssets Metadata (max_values: None, max_size: Some(738), added: 3213, mode: MaxEncodedLen)
	/// The range of component `n` is `[0, 50]`.
	/// The range of component `s` is `[0, 50]`.
	/// The range of component `n` is `[0, 50]`.
	/// The range of component `s` is `[0, 50]`.
	fn force_set_metadata(_n: u32, s: u32, ) -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `81`
		//  Estimated: `4273`
		// Minimum execution time: 14_186_000 picoseconds.
		Weight::from_parts(14_717_332, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			// Standard Error: 517
			.saturating_add(Weight::from_parts(2_595, 0).saturating_mul(s.into()))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:0)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Metadata` (r:1 w:1)
	/// Proof: `ForeignAssets::Metadata` (`max_values`: None, `max_size`: Some(738), added: 3213, mode: `MaxEncodedLen`)
	fn force_clear_metadata() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `406`
		//  Estimated: `4273`
		// Minimum execution time: 29_499_000 picoseconds.
		Weight::from_parts(29_918_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn force_asset_status() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 13_815_000 picoseconds.
		Weight::from_parts(14_138_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(1))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Approvals` (r:1 w:1)
	/// Proof: `ForeignAssets::Approvals` (`max_values`: None, `max_size`: Some(746), added: 3221, mode: `MaxEncodedLen`)
	fn approve_transfer() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `276`
		//  Estimated: `4273`
		// Minimum execution time: 33_029_000 picoseconds.
		Weight::from_parts(33_524_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Approvals` (r:1 w:1)
	/// Proof: `ForeignAssets::Approvals` (`max_values`: None, `max_size`: Some(746), added: 3221, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:2 w:2)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `System::Account` (r:1 w:1)
	/// Proof: `System::Account` (`max_values`: None, `max_size`: Some(128), added: 2603, mode: `MaxEncodedLen`)
	fn transfer_approved() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `520`
		//  Estimated: `7404`
		// Minimum execution time: 63_205_000 picoseconds.
		Weight::from_parts(64_078_000, 0)
			.saturating_add(Weight::from_parts(0, 7404))
			.saturating_add(T::DbWeight::get().reads(5))
			.saturating_add(T::DbWeight::get().writes(5))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Approvals` (r:1 w:1)
	/// Proof: `ForeignAssets::Approvals` (`max_values`: None, `max_size`: Some(746), added: 3221, mode: `MaxEncodedLen`)
	fn cancel_approval() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `446`
		//  Estimated: `4273`
		// Minimum execution time: 34_948_000 picoseconds.
		Weight::from_parts(35_484_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Approvals` (r:1 w:1)
	/// Proof: `ForeignAssets::Approvals` (`max_values`: None, `max_size`: Some(746), added: 3221, mode: `MaxEncodedLen`)
	fn force_cancel_approval() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `446`
		//  Estimated: `4273`
		// Minimum execution time: 35_722_000 picoseconds.
		Weight::from_parts(36_266_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn set_min_balance() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 15_855_000 picoseconds.
		Weight::from_parts(16_182_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(1))
			.saturating_add(T::DbWeight::get().writes(1))
	}
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `System::Account` (r:1 w:1)
	/// Proof: `System::Account` (`max_values`: None, `max_size`: Some(128), added: 2603, mode: `MaxEncodedLen`)
	fn touch() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `345`
		//  Estimated: `4273`
		// Minimum execution time: 34_984_000 picoseconds.
		Weight::from_parts(35_512_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(3))
			.saturating_add(T::DbWeight::get().writes(3))
	}
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn touch_other() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `242`
		//  Estimated: `4273`
		// Minimum execution time: 33_041_000 picoseconds.
		Weight::from_parts(34_124_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `System::Account` (r:1 w:1)
	/// Proof: `System::Account` (`max_values`: None, `max_size`: Some(128), added: 2603, mode: `MaxEncodedLen`)
	fn refund() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `471`
		//  Estimated: `4273`
		// Minimum execution time: 31_728_000 picoseconds.
		Weight::from_parts(32_012_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(3))
			.saturating_add(T::DbWeight::get().writes(3))
	}
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Asset` (r:1 w:1)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	fn refund_other() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `401`
		//  Estimated: `4273`
		// Minimum execution time: 29_432_000 picoseconds.
		Weight::from_parts(29_968_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}
	/// Storage: `ForeignAssets::Asset` (r:1 w:0)
	/// Proof: `ForeignAssets::Asset` (`max_values`: None, `max_size`: Some(808), added: 3283, mode: `MaxEncodedLen`)
	/// Storage: `ForeignAssets::Account` (r:1 w:1)
	/// Proof: `ForeignAssets::Account` (`max_values`: None, `max_size`: Some(732), added: 3207, mode: `MaxEncodedLen`)
	fn block() -> Weight {
		// Proof Size summary in bytes:
		//  Measured:  `350`
		//  Estimated: `4273`
		// Minimum execution time: 18_827_000 picoseconds.
		Weight::from_parts(19_172_000, 0)
			.saturating_add(Weight::from_parts(0, 4273))
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(1))
	}
}
