<script lang="ts">
	import { colors, utils } from './utils';
	import Items from './items.svelte';
	import Stars from './stars.svelte';

	export let champion: Champion | null | undefined = null;
</script>

<div class="wrapper">
	{#if champion}
		<div class="star-wrapper">
			<Stars stars={champion.stars} />
		</div>
		<div class="item-wrapper">
			<Items items={champion.items} />
		</div>
		<div
			class="bench-champion border-4"
			style="background-image: url({utils.getIconImage(
				champion
			)}); border-color: {utils.getCostColor(champion)};"
		></div>
		{#if champion.chosen}
			<div class="chosen" style="background-image: url({utils.getChosenImage(champion)})" />
			<div class="champion-name text-xs font-sans text-center" style="color: {colors.chosenColor}">
				{champion.name}
			</div>
		{:else}
			<div class="champion-name text-xs font-sans text-center">
				{champion.name}
			</div>
		{/if}
	{:else}
		<div class="bench-champion border-4 bg-base-200 border-base-300"></div>
		<div class="chosen" />
	{/if}
</div>

<style>
	@import './champion.css';
	.bench-champion {
		width: var(--bench-size);
		height: var(--bench-size);
		background-size: var(--bench-size);
		background-position: center;
	}
</style>
