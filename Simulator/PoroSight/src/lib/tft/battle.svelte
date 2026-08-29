<script lang="ts">
	import ChampionBattleHex from './champion/championBattleHex.svelte';
	import { locationTo2DIndex, computeRoundString } from './util';

	export let battle: Battle;
	export let currentPlayerID: number;

	$: boardComponents = createBoard(battle.board);
	$: opponentBoardComponents = createReversedBoard(battle.opponentBoard);
	$: [ourOutcome, oppOutcome] = outcome(battle);
	$: roundString = computeRoundString(battle.round);

	const outcome = (battle: Battle) => {
		if (battle.result === 'win') {
			return ['W', 'L'];
		} else if (battle.result === 'loss') {
			return ['L', 'W'];
		} else {
			return ['T', 'T'];
		}
	};

	const outcomeColor = (outcome: string) => {
		if (outcome === 'W') {
			return 'text-green-500';
		} else if (outcome === 'L') {
			return 'text-red-500';
		} else {
			return 'text-gray-500';
		}
	};

	const emptyBoard = () => {
		let components = [];
		for (let i = 0; i < 4; i++) {
			let row = [];
			for (let j = 0; j < 7; j++) {
				row.push({
					component: ChampionBattleHex,
					props: { champion: null }
				});
			}
			components.push(row);
		}

		return components;
	};

	const createBoard = (b: Champion[]) => {
		let components = emptyBoard();

		for (const champion of b) {
			if (champion) {
				let [x, y] = locationTo2DIndex(champion.location);
				components[x][y] = {
					component: ChampionBattleHex,
					props: { champion: champion }
				};
			}
		}

		return components;
	};

	const createReversedBoard = (b: Champion[] | null) => {
		if (!b) {
			return emptyBoard();
		}

		let components = createBoard(b);
		let reversed = components.reverse();

		for (let i = 0; i < reversed.length; i++) {
			reversed[i] = reversed[i].reverse();
		}

		return reversed;
	};
</script>

<div class="flex flex-col items-center flex-initial">
	<div class="align-middle text-lg stat-value">Round {roundString}</div>

	<div class="board align-middle">
		<div class="hex-grid">
			<div class="battle-hex-row first reversed">
				{#each opponentBoardComponents[0] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
			<div class="battle-hex-row second reversed">
				{#each opponentBoardComponents[1] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
			<div class="battle-hex-row third reversed">
				{#each opponentBoardComponents[2] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
			<div class="battle-hex-row fourth reversed">
				{#each opponentBoardComponents[3] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
		</div>
	</div>

	<div class="flex w-full">
		<div class="stat place-items-center mt-3">
			<div class="stat-desc">Damage: 0</div>
			<div class="stat-value text-md {outcomeColor(ourOutcome)}">{ourOutcome}</div>
			<div class="stat-title text-sm capitalize">
				<span class="capitalize">{`Player_${currentPlayerID}`}</span> (us)
			</div>
		</div>
		<div class="divider divider-horizontal text-accent">VS</div>
		<div class="stat place-items-center mb-3">
			<div class="stat-title text-sm"><span class="capitalize">{battle.opponent}</span> (opp)</div>
			<div class="stat-value text-md {outcomeColor(oppOutcome)}">{oppOutcome}</div>
			<div class="stat-desc">Damage: -2</div>
		</div>
	</div>

	<div class="board align-middle">
		<div class="hex-grid">
			<div class="battle-hex-row first">
				{#each boardComponents[0] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
			<div class="battle-hex-row second">
				{#each boardComponents[1] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
			<div class="battle-hex-row third">
				{#each boardComponents[2] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
			<div class="battle-hex-row fourth">
				{#each boardComponents[3] as c, _}
					<svelte:component this={c.component} {...c.props} />
				{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.board {
		padding-right: var(--board-padding);
	}
	.hex-grid {
		font-size: 0;
	}

	.battle-hex-row {
		position: sticky;
	}

	.battle-hex-row.first {
		z-index: 4;
	}
	.battle-hex-row.second {
		z-index: 3;
		transform: translate(var(--battle-row-translateX), var(--battle-row-translateY));
	}
	.battle-hex-row.third {
		z-index: 2;
		transform: translate(0, calc(var(--battle-row-translateY) * 2));
	}
	.battle-hex-row.fourth {
		z-index: 1;
		transform: translate(var(--battle-row-translateX), calc(var(--battle-row-translateY) * 3));
	}

	.battle-hex-row.first.reversed {
		z-index: 4;
		transform: translate(var(--battle-row-translateX), calc(var(--battle-row-translateY) * -3));
	}

	.battle-hex-row.second.reversed {
		z-index: 3;
		transform: translate(0, calc(var(--battle-row-translateY) * -2));
	}

	.battle-hex-row.third.reversed {
		z-index: 2;
		transform: translate(var(--battle-row-translateX), calc(var(--battle-row-translateY) * -1));
	}

	.battle-hex-row.fourth.reversed {
		z-index: 1;
		transform: translate(0, 0);
	}
</style>
