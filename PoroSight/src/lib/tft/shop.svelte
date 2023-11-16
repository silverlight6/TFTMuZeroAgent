<script lang="ts">
	import ChampionShop from './championShop.svelte';
	import { currentState, currentDiff } from '$lib/state';

	let shop: Shop = ([] = createEmptyShop());
	let shopComponents = createShopComponents(shop);

	function createEmptyShop() {
		return Array(5).fill(null);
	}

	function createShop(s) {
		let shop = createEmptyShop();
		for (let i = 0; i < s.length; i++) {
			shop[i] = s[i];
		}
		return shop;
	}

	function createShopComponents(s) {
		let components = [];
		for (const champion of s) {
			components.push({
				component: ChampionShop,
				props: { champion }
			});
		}
		return components;
	}

	$: if (currentState) {
		shop = createShop($currentState.shop);
		shopComponents = createShopComponents(shop);
	} else {
		shop = createEmptyShop();
		shopComponents = createShopComponents(shop);
	}
</script>

<div class="flex gap-3 m-3">
	{#each shopComponents as c, index}
		<svelte:component this={c.component} {...c.props} />
	{/each}
</div>
