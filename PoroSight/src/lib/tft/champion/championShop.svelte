<script lang="ts">
	import { colors, utils } from './utils';

	export let champion: Champion | null | undefined = null;

	// 216 x 175
</script>

<div class="wrapper">
	{#if champion}
		<img
			class="shop-img border-4"
			style="border-color: {utils.getCostColor(champion)};"
			src={utils.getShopImage(champion)}
			alt={champion.name}
		/>
		<div class="champion-name text-xs font-sans text-center">
			{champion.name}
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
		<div class="shop-img" style="background-color: {colors.defaultBg}"></div>
	{/if}
</div>

<style>
	@import './champion.css';
	.shop-img {
		width: calc(var(--shop-width) * var(--shop-scale));
		height: calc(var(--shop-height) * var(--shop-scale));
	}
</style>
