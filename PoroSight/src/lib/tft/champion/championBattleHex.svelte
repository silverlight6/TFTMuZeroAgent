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
		<div class="battle-hexagon-wrapper" style="background-color: {utils.getCostColor(champion)};">
			<div class="battle-hexagon" style="background-image: url({utils.getIconImage(champion)})" />
		</div>
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
		<div class="star-wrapper" />
		<div class="item-wrapper" />
		<div class="battle-hexagon-wrapper bg-base-300">
			<div class="battle-hexagon bg-base-200"></div>
		</div>
		<div class="chosen" />
	{/if}
</div>

<style>
	@import './champion.css';
	.battle-hexagon-wrapper {
		width: var(--battle-hex-width);
		height: var(--battle-hex-height);
		margin: var(--battle-hex-margin);

		display: inline-block;
		font-size: initial;
		clip-path: var(--hex-polygon);
	}

	.battle-hexagon {
		width: calc(var(--battle-hex-width) * var(--hex-inner-scale));
		height: calc(var(--battle-hex-height) * var(--hex-inner-scale));
		margin: 0 auto;

		display: inline-block;
		font-size: initial;
		clip-path: var(--hex-polygon);
		background-size: calc(var(--w) * (var(--hex-inner-scale) * 0.9));
		background-position: center;

		position: absolute;
		left: 50%;
		top: 50%;
		transform: translate(-50%, -50%);
	}
</style>
