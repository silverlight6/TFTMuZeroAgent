<script lang="ts">
	import ChampionHex from './champion/championHex.svelte';

	export let board: Champion[] = [];

	$: boardComponents = createBoard(board);

	const emptyBoard = () => {
		let components = [];
		for (let i = 0; i < 4; i++) {
			let row = [];
			for (let j = 0; j < 7; j++) {
				row.push({
					component: ChampionHex,
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
					component: ChampionHex,
					props: { champion: champion }
				};
			}
		}

		return components;
	};

	function locationTo2DIndex(location: number): [number, number] {
		const xLoc: number = Math.floor(location / 4);
		const yLoc: number = location - xLoc * 4;

		// This is in the (7, 4) format that the sim uses.
		// We need to rotate it to the (4, 7) format that the board uses.
		// (0, 3) -> (0, 0), (0, 2) -> (1, 0) ...
		const x: number = 3 - yLoc;
		const y: number = xLoc;

		return [x, y];
	}
</script>

<!-- The board is a 4x7 grid of hexagons -->
<!-- The hexagons are staggered -->

<div class="board align-middle">
	<div class="hex-grid">
		<div class="hex-row first">
			{#each boardComponents[0] as c, _}
				<svelte:component this={c.component} {...c.props} />
			{/each}
		</div>
		<div class="hex-row second">
			{#each boardComponents[1] as c, _}
				<svelte:component this={c.component} {...c.props} />
			{/each}
		</div>
		<div class="hex-row third">
			{#each boardComponents[2] as c, _}
				<svelte:component this={c.component} {...c.props} />
			{/each}
		</div>
		<div class="hex-row fourth">
			{#each boardComponents[3] as c, _}
				<svelte:component this={c.component} {...c.props} />
			{/each}
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

	.hex-row {
		position: sticky;
	}

	.hex-row.first {
		z-index: 4;
	}
	.hex-row.second {
		z-index: 3;
		transform: translate(var(--row-translateX), var(--row-translateY));
	}
	.hex-row.third {
		z-index: 2;
		transform: translate(0, calc(var(--row-translateY) * 2));
	}
	.hex-row.fourth {
		z-index: 1;
		transform: translate(var(--row-translateX), calc(var(--row-translateY) * 3));
	}
</style>
