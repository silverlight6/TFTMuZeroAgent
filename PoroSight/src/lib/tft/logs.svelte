<script lang="ts">
	import { currentState, currentDiff } from '$lib/state';
	import { boardLocationToIndex, createEmptyItemBench, createItemBench } from '$lib/util';
	import { get } from 'svelte/store';

	let log: string[] = [];

	function createLog(currentState: PlayerState, diff: PlayerDiff) {
		let action = diff.action;

		if (!action) {
			let boardString = getChampionString(currentState.board);
			let benchString = getChampionString(currentState.bench);
			let itemString = getItemString(currentState.items);
			return [
				`Start of game;`,
				`Gained ${boardString} on board;`,
				`Gained ${benchString} on bench;`,
				`Gained ${itemString} on item bench;`
			];
		}

		let newLog = [`${action}`];
		let [action_type, from, to] = action;
		console.log(diff);

		// Move Action
		if (action_type == 5) {
			let from_board = from < 28;
			let to_board = to < 28;

			if (from_board && to_board) {
				for (const champion of diff.board) {
					if (champion) {
						let f = champion.location == from ? to : from;
						let t = champion.location == from ? from : to;
						newLog.push(
							`Moved ${champion.name} from ${boardLocationToIndex(f)} to ${boardLocationToIndex(
								t
							)};`
						);
					}
				}
			} else if (from_board && !to_board) {
				for (const champion of diff.board) {
					if (champion) {
						newLog.push(
							`Moved ${champion.name} from bench (${to}) to board (${boardLocationToIndex(from)});`
						);
					}
				}
				for (const champion of diff.bench) {
					if (champion) {
						newLog.push(
							`Moved ${champion.name} from board (${boardLocationToIndex(from)}) to bench (${to})`
						);
					}
				}
			} else if (!from_board && to_board) {
				for (const champion of diff.bench) {
					if (champion) {
						newLog.push(
							`Moved ${champion.name} from bench (${from}) to board (${boardLocationToIndex(to)});`
						);
					}
				}
				for (const champion of diff.board) {
					if (champion) {
						newLog.push(
							`Moved ${champion.name} from board (${boardLocationToIndex(to)}) to bench (${from})`
						);
					}
				}
			} else {
				for (const champion of diff.bench) {
					if (champion) {
						let f = champion.location == from ? to : from;
						let t = champion.location == from ? from : to;
						newLog.push(`Moved ${champion.name} from ${f} to ${t};`);
					}
				}
			}
		}

		return newLog;
	}

	function getChampionString(champions: Champion[]) {
		let championString = '';
		for (const champion of champions) {
			championString += `${champion.name} w/ ${champion.stars} star(s) w/ {${getItemString(
				champion.items
			)}}; `;
		}

		if (!championString) {
			return 'nothing';
		}

		return championString;
	}

	function getItemString(items: Item[]) {
		let itemString = '';
		for (const item of items) {
			itemString += `${item.name};`;
		}

		if (!itemString) {
			return 'nothing';
		}

		return itemString;
	}

	$: if (currentDiff) {
		log = createLog($currentState, $currentDiff);
	} else {
		log = [];
	}
</script>

{#each log as line}
	<p class="text-sm text-base-content">{line}</p>
{/each}
