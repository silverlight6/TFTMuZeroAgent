<script lang="ts">
	import ChampionShop from './champion/championShop.svelte';

	export let shop: Champion[] = [];

	$: shopComponents = createShop(shop);

	const emptyShop = () => {
		return Array(5).fill({ component: ChampionShop, props: { champion: null } });
	};

	const createShop = (shop: Champion[]) => {
		let components = emptyShop();

		for (const champion of shop) {
			if (champion) {
				let index = champion.location;
				components[index] = {
					component: ChampionShop,
					props: { champion }
				};
			}
		}

		return components;
	};
</script>

<div class="flex gap-3 m-3">
	{#each shopComponents as c, index}
		<svelte:component this={c.component} {...c.props} />
	{/each}
</div>
