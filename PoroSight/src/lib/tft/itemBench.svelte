<script lang="ts">
	import { getItemImage } from '$lib/image';

	export let items: Item[] = [];

	$: itemComponents = createItems(items);

	function createItems(items: Item[]) {
		let components = Array(9).fill(null);

		for (let i = 0; i < items.length; i++) {
			if (items[i]) {
				components[i] = {
					image: getItemImage(items[i].name),
					name: items[i].name
				};
			}
		}

		return components;
	}
</script>

<div class="itemBench flex gap-2 m-3">
	{#each itemComponents as item, _}
		<div class="flex flex-col gap-1">
			{#if item}
				<img class="itemBenchItem" src={item.image} alt={item.name} />
			{:else}
				<div class="itemBenchItem bg-base-300"></div>
			{/if}
		</div>
	{/each}
</div>

<style>
	.itemBench {
		margin-top: var(--item-bench-margin);
	}
	.itemBenchItem {
		width: var(--item-bench-size);
		height: var(--item-bench-size);
	}
</style>
