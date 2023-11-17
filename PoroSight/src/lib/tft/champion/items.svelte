<script lang="ts">
	import { utils } from './utils';

	export let items: Item[] = [];

	$: itemImages = utils.getItemImages(items);
</script>

{#each itemImages as itemImage, _}
	<div class="item inline-block" style="background-image: url({itemImage})" />
{/each}

<style>
	.item {
		width: var(--image-size);
		height: var(--image-size);
		background-size: var(--image-size);
	}
</style>
