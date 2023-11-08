// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg(test)]

use crate::{mock::*, StakeImbalance};

use frame_election_provider_support::SortedListProvider;
use frame_support::assert_ok;
use sp_staking::{OnStakingUpdate, Stake, StakingInterface};

// keeping tests clean.
type A = AccountId;
type B = Balance;

#[test]
fn setup_works() {
	ExtBuilder::default().build_and_execute(|| {
		assert!(TestNominators::get().is_empty());
		assert_eq!(VoterBagsList::count(), 0);

		assert!(TestValidators::get().is_empty());
		assert_eq!(TargetBagsList::count(), 0);
	});

	ExtBuilder::default().populate_lists().build_and_execute(|| {
		assert!(!TestNominators::get().is_empty());
		assert_eq!(VoterBagsList::count(), 4); // voter list has 2x nominatiors + 2x validators

		assert!(!TestValidators::get().is_empty());
		assert_eq!(TargetBagsList::count(), 2);
	});
}

#[test]
fn update_score_works() {
	ExtBuilder::default().populate_lists().build_and_execute(|| {
		assert!(VoterBagsList::contains(&1));
		assert_eq!(VoterBagsList::get_score(&1), Ok(100));

		crate::Pallet::<Test>::update_score::<VoterBagsList>(&1, StakeImbalance::Negative(10));
		assert_eq!(VoterBagsList::get_score(&1), Ok(90));

		crate::Pallet::<Test>::update_score::<VoterBagsList>(&1, StakeImbalance::Positive(100));
		assert_eq!(VoterBagsList::get_score(&1), Ok(190));

		// when score decreases to 0, the node is not removed automatically and its balance is 0.
		let current_score = VoterBagsList::get_score(&1).unwrap();
		crate::Pallet::<Test>::update_score::<VoterBagsList>(
			&1,
			StakeImbalance::Negative(current_score),
		);
		assert!(VoterBagsList::contains(&1));
		assert_eq!(VoterBagsList::get_score(&1), Ok(0));
	})
}

#[test]
#[should_panic = "Defensive failure has been triggered!: \"`update_score` on non-existant staker\": 1"]
fn update_score_non_existing_defensive_works() {
	ExtBuilder::default().build_and_execute(|| {
		assert!(!VoterBagsList::contains(&1));
		// not expected to update score of a non-existing staker.
		crate::Pallet::<Test>::update_score::<VoterBagsList>(&1, StakeImbalance::Positive(100));
	});
}

#[test]
#[should_panic]
fn update_score_below_zero_defensive_works() {
	ExtBuilder::default().populate_lists().build_and_execute(|| {
		assert!(VoterBagsList::contains(&1));
		assert_eq!(VoterBagsList::get_score(&1), Ok(100));
		// updating the score below 0 is unexpected.
		crate::Pallet::<Test>::update_score::<VoterBagsList>(&1, StakeImbalance::Negative(500));
	})
}

// same as test above but does not panic after defensive so we can test invariants.
#[test]
#[cfg(not(debug_assertions))]
fn update_score_below_zero_defensive_no_panic_works() {
	ExtBuilder::default().populate_lists().build_and_execute(|| {
		assert!(VoterBagsList::contains(&1));
		assert_eq!(VoterBagsList::get_score(&1), Ok(100));
		// updating the score below 0 is unexpected and saturates to 0.
		crate::Pallet::<Test>::update_score::<VoterBagsList>(&1, StakeImbalance::Negative(500));
		assert!(VoterBagsList::contains(&1));
		assert_eq!(VoterBagsList::get_score(&1), Ok(0));

		let n = TestNominators::get();
		assert!(n.get(&1).is_some());
	})
}

#[test]
fn on_stake_update_works() {
	//TODO(gpestana)
}

#[test]
fn on_nominator_add_works() {
	ExtBuilder::default().build_and_execute(|| {
		let n = TestNominators::get();
		assert!(!VoterBagsList::contains(&5));
		assert_eq!(n.get(&5), None);

		// add 5 as staker.
		TestNominators::mutate(|n| {
			n.insert(5, Default::default());
		});

		<StakeTracker as OnStakingUpdate<A, B>>::on_nominator_add(&5);
		assert!(VoterBagsList::contains(&5));
	})
}

#[test]
fn on_validator_add_works() {
	ExtBuilder::default().build_and_execute(|| {
		let n = TestNominators::get();
		let v = TestValidators::get();
		assert!(!VoterBagsList::contains(&5));
		assert!(!TargetBagsList::contains(&5));
		assert!(n.get(&5).is_none() && v.get(&5).is_none());

		// add 5 as staker (target and voter).
		TestNominators::mutate(|n| {
			n.insert(5, Default::default());
		});
		TestValidators::mutate(|n| {
			n.insert(5, Default::default());
		});
	})
}

#[test]
#[should_panic = "Defensive failure has been triggered!: Duplicate: \"staker should not exist in VoterList, as per the contract with staking.\""]
fn on_nominator_add_already_exists_defensive_works() {
	ExtBuilder::default().populate_lists().build_and_execute(|| {
		// voter already exists in the list, trying to emit `on_add_nominator` again will fail.
		assert!(VoterBagsList::contains(&1));
		<StakeTracker as OnStakingUpdate<A, B>>::on_nominator_add(&1);
	});
}

#[test]
#[should_panic = "Defensive failure has been triggered!: Duplicate: \"staker should not exist in TargetList, as per the contract with staking.\""]
fn on_validator_add_already_exists_defensive_works() {
	ExtBuilder::default().populate_lists().build_and_execute(|| {
		// validator already exists in the list, trying to emit `on_add_validator` again will fail.
		assert!(TargetBagsList::contains(&10));
		<StakeTracker as OnStakingUpdate<A, B>>::on_validator_add(&10);
	});
}

#[test]
fn on_nominator_remove_works() {
	ExtBuilder::default().populate_lists().build_and_execute(|| {
		assert!(VoterBagsList::contains(&1));
		let nominator_score = VoterBagsList::get_score(&1).unwrap();

		let nominations = <StakingMock as StakingInterface>::nominations(&1).unwrap();
		assert!(nominations.len() == 1);
		let nomination_score_before = TargetBagsList::get_score(&nominations[0]).unwrap();

		<StakeTracker as OnStakingUpdate<A, B>>::on_nominator_remove(&1, nominations.clone());

		// the nominator was removed from the voter list.
		assert!(!VoterBagsList::contains(&1));

		// now, the score of the nominated by 1 has less `nominator_score` stake than before the
		// nominator was removed.
		let nomination_score_after = TargetBagsList::get_score(&nominations[0]).unwrap();
		assert!(nomination_score_after == nomination_score_before - nominator_score);
	})
}

#[test]
#[should_panic = "Defensive failure has been triggered!: NodeNotFound: \"the nominator exists in the list as per the contract with staking; qed.\""]
fn on_nominator_remove_defensive_works() {
	ExtBuilder::default().populate_lists().build_and_execute(|| {
		assert!(VoterBagsList::contains(&1));

		// remove 1 from the voter list to check if the defensive is triggered in the next call,
		// while maintaining it as a staker so it does not early exist at the staking mock
		// implementation.
		assert_ok!(VoterBagsList::on_remove(&1));

		<StakeTracker as OnStakingUpdate<A, B>>::on_nominator_remove(&1, vec![]);
	})
}

#[test]
#[should_panic = "Defensive failure has been triggered!: NodeNotFound: \"the validator exists in the list as per the contract with staking; qed.\""]
fn on_validator_remove_defensive_works() {
	ExtBuilder::default().build_and_execute(|| {
		assert!(!TargetBagsList::contains(&1));
		<StakeTracker as OnStakingUpdate<A, B>>::on_validator_remove(&1);
	})
}

#[test]
fn on_nominator_update_works() {
	// TODO(gpestana)
}

mod staking_integration {
	use super::*;

	#[test]
	fn staking_interface_works() {
		ExtBuilder::default().build_and_execute(|| {
			assert_eq!(TestNominators::get().iter().count(), 0);
			assert_eq!(TestValidators::get().iter().count(), 0);

			add_nominator(1, 100);
			let n = TestNominators::get();
			assert_eq!(n.get(&1).unwrap().0, Stake { active: 100u64, total: 100u64 });

			add_validator(2, 200);
			let v = TestValidators::get();
			assert_eq!(v.get(&2).copied().unwrap(), Stake { active: 200u64, total: 200u64 });
		})
	}

	#[test]
	fn on_add_stakers_works() {
		ExtBuilder::default().build_and_execute(|| {
			add_nominator(1, 100);
			assert_eq!(TargetBagsList::count(), 0);
			assert_eq!(VoterBagsList::count(), 1);
			assert_eq!(VoterBagsList::get_score(&1).unwrap(), 100);

			add_validator(10, 200);
			assert_eq!(VoterBagsList::count(), 2); // 1x nominator + 1x validator
			assert_eq!(TargetBagsList::count(), 1);
			assert_eq!(TargetBagsList::get_score(&10).unwrap(), 200);
		})
	}

	#[test]
	fn on_update_stake_works() {
		ExtBuilder::default().build_and_execute(|| {
			add_nominator(1, 100);
			assert_eq!(VoterBagsList::get_score(&1).unwrap(), 100);
			update_stake(1, 200, stake_of(1));
			assert_eq!(VoterBagsList::get_score(&1).unwrap(), 200);

			add_validator(10, 100);
			assert_eq!(TargetBagsList::get_score(&10).unwrap(), 100);
			update_stake(10, 200, stake_of(10));
			assert_eq!(TargetBagsList::get_score(&10).unwrap(), 200);
		})
	}

	#[test]
	fn on_remove_stakers_works() {
		ExtBuilder::default().build_and_execute(|| {
			add_nominator(1, 100);
			assert!(VoterBagsList::contains(&1));
			remove_staker(1);
			assert!(!VoterBagsList::contains(&1));

			add_validator(10, 100);
			assert!(TargetBagsList::contains(&10));
			remove_staker(10);
			assert!(!TargetBagsList::contains(&10));
		})
	}

	#[test]
	fn on_remove_stakers_with_nominations_works() {
		ExtBuilder::default().populate_lists().build_and_execute(|| {
			assert_eq!(get_scores::<TargetBagsList>(), vec![(10, 300), (11, 200)]);

			assert!(VoterBagsList::contains(&1));
			assert_eq!(VoterBagsList::get_score(&1), Ok(100));
			assert_eq!(TargetBagsList::get_score(&10), Ok(300));

			// remove nominator deletes node from voter list and updates the stake of its
			// nominations.
			remove_staker(1);
			assert!(!VoterBagsList::contains(&1));
			assert_eq!(TargetBagsList::get_score(&10), Ok(200));
		})
	}

	#[test]
	fn on_nominator_update_works() {
		ExtBuilder::default().populate_lists().build_and_execute(|| {
			assert_eq!(
				get_scores::<VoterBagsList>(),
				vec![(10, 100), (11, 100), (1, 100), (2, 100)]
			);
			assert_eq!(get_scores::<TargetBagsList>(), vec![(10, 300), (11, 200)]);

			add_validator(20, 50);
			// removes nomination from 10 and adds nomination to new validator, 20.
			update_nominations_of(2, vec![11, 20]);

			// new voter (validator) 2 with 100 stake. note that the voter score is not updated
			// automatically.
			assert_eq!(
				get_scores::<VoterBagsList>(),
				vec![(10, 100), (11, 100), (1, 100), (2, 100), (20, 50)]
			);

			// target list has been updated:
			// -100 score for 10
			// +100 score for 11
			// +100 score for 20
			assert_eq!(get_scores::<TargetBagsList>(), vec![(10, 200), (11, 200), (20, 150)]);
		})
	}
}
