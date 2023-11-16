<script lang="ts">
	import { getItemImage } from '$lib/image';
	import { createEmptyItemBench, createItemBench } from '$lib/util';
	import { currentState, currentDiff } from '$lib/state';

	let itemBench: ItemBench = createEmptyItemBench();

	$: if (currentState) {
		itemBench = createItemBench($currentState.items);
	} else {
		itemBench = createEmptyItemBench();
	}
</script>

<div class="flex gap-2 m-3">
	{#each itemBench as item, index}
		<div class="flex flex-col gap-1">
			{#if item}
				<img class="w-12 h-12" src={getItemImage(item.name)} alt={item.name} />
			{:else}
				<div class="w-12 h-12 bg-base-content"></div>
			{/if}
		</div>
	{/each}
</div>
