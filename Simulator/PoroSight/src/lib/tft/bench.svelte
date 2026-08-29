<script lang="ts">
	import ChampionSquare from './champion/championSquare.svelte';

	export let bench: Champion[] = [];

	$: benchComponents = createBench(bench);

	const emptyBench = () => {
		return Array(9).fill({
			component: ChampionSquare,
			props: { champion: null }
		});
	};

	const createBench = (bench: Champion[]) => {
		let components = emptyBench();

		for (const champion of bench) {
			if (champion) {
				let index = champion.location;
				components[index] = {
					component: ChampionSquare,
					props: { champion }
				};
			}
		}

		return components;
	};
</script>

<div class="flex gap-3 m-3">
	{#each benchComponents as c, index}
		<svelte:component this={c.component} {...c.props} />
	{/each}
</div>
