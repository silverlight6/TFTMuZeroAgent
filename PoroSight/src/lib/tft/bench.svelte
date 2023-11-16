<script lang="ts">
	import ChampionSquare from './championSquare.svelte';
	import { createEmptyBench, createBench } from '$lib/util';
	import { currentState, currentDiff } from '$lib/state';

	let bench: Bench = createEmptyBench();
	let benchComponents = createBenchComponents(bench);

	function createBenchComponents(b) {
		let components = [];
		for (const champion of b) {
			components.push({
				component: ChampionSquare,
				props: { champion }
			});
		}
		return components;
	}

	$: if (currentState) {
		bench = createBench($currentState.bench);
		benchComponents = createBenchComponents(bench);
	} else {
		bench = createEmptyBench();
		benchComponents = createBenchComponents(bench);
	}
</script>

<div class="flex gap-3 m-3">
	{#each benchComponents as c, index}
		<svelte:component this={c.component} {...c.props} />
	{/each}
</div>

<style>
	.hex-row {
		--m: 4px;
		--w: 70px;
		--h: calc(var(--w) * 0.866);
		--s: calc(var(--w) / 2);
		--tx: calc(calc(var(--h) / 2) + var(--m));
		--ty: calc(calc(var(--w) - var(--s)) / -2);

		position: sticky;
	}
</style>
