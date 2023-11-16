<script lang="ts">
	import { getChampionShopImage, getChosenImage, getTraitImage } from '$lib/image';
	import { getCostColor } from '$lib/util';

	export let champion: Champion | null | undefined = null;

	let defaultColor: string = 'bg-base-content';
	let defaultBorder: string = 'border-primary';

	let cost: number | null = null;
	let chosen: boolean | string = false;

	let championImage: string | null = null;
	// let traitImages: string[] = [];

	let costColor = getCostColor(cost);
	let chosenImage = getChosenImage(chosen);

	function updateImage(champion: Champion | null) {
		if (!champion) {
			championImage = null;
			cost = null;
			chosen = false;
		} else {
			championImage = getChampionShopImage(champion.name);
			// traitImages = champion.traits.map((trait) => getTraitImage(trait.name));
			cost = champion.cost;
			chosen = champion.chosen;
		}
		costColor = getCostColor(cost);
		chosenImage = getChosenImage(chosen);
	}

	// Call the function whenever champion changes
	$: updateImage(champion);
	// 216 x 175
</script>

<div class="">
	{#if champion}
		<img class="shop-img border-4 border-[{costColor}]" src={championImage} alt={champion.name} />
	{:else}
		<div class="shop-img bg-base-content"></div>
	{/if}
</div>

<style>
	.shop-img {
		--width: 216px;
		--height: 175px;

		width: calc(var(--width) * 0.5);
		height: calc(var(--height) * 0.5);
	}
</style>
