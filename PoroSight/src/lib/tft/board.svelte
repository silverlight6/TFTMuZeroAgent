<script lang="ts">
	import ChampionHex from './championHex.svelte';
	import { createEmptyBoard, createBoard } from '$lib/util';
	import { currentState, currentDiff } from '$lib/state';

	let board: Board = createEmptyBoard();
	let boardComponents = createBoardComponents(board);

	function createBoardComponents(b) {
		let components = [];
		for (const row of b) {
			let rowComponents = [];
			for (const champion of row) {
				rowComponents.push({
					component: ChampionHex,
					props: { champion }
				});
			}
			components.push(rowComponents);
		}
		return components;
	}

	$: if (currentState) {
		board = createBoard($currentState.board);
		boardComponents = createBoardComponents(board);
	} else {
		board = createEmptyBoard();
		boardComponents = createBoardComponents(board);
	}
</script>

<!-- The board is a 4x7 grid of hexagons -->
<!-- The hexagons are staggered -->

<div class="m-3 align-middle">
	<div class="hex-grid">
		<div class="hex-row first">
			{#each boardComponents[0] as c, index}
				<svelte:component this={c.component} {...c.props} />
			{/each}
		</div>
		<div class="hex-row second">
			{#each boardComponents[1] as c, index}
				<svelte:component this={c.component} {...c.props} />
			{/each}
		</div>
		<div class="hex-row third">
			{#each boardComponents[2] as c, index}
				<svelte:component this={c.component} {...c.props} />
			{/each}
		</div>
		<div class="hex-row fourth">
			{#each boardComponents[3] as c, index}
				<svelte:component this={c.component} {...c.props} />
			{/each}
		</div>
	</div>
</div>

<style>
	.hex-grid {
		font-size: 0;
	}

	.hex-row {
		--m: 4px;
		--w: 70px;
		--h: calc(var(--w) * 0.866);
		--s: calc(var(--w) / 2);
		--tx: calc(calc(var(--h) / 2) + var(--m));
		--ty: calc(calc(var(--w) - var(--s)) / -2);

		position: sticky;
	}

	.hex-row.first {
		z-index: 4;
	}
	.hex-row.second {
		z-index: 3;
		transform: translate(var(--tx), var(--ty));
	}
	.hex-row.third {
		z-index: 2;
		transform: translate(0, calc(var(--ty) * 2));
	}
	.hex-row.fourth {
		z-index: 1;
		transform: translate(var(--tx), calc(var(--ty) * 3));
	}
</style>
