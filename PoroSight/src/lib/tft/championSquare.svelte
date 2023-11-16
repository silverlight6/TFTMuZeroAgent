<script lang="ts">
	import { getChampionIconImage, getItemImage, getChosenImage } from '$lib/image';
	import { getCostColor, getStars } from '$lib/util';

	export let champion: Champion | null | undefined = null;

	let defaultColor: string = 'bg-base-content';
	let defaultBorder: string = 'border-primary';

	let championImage: string | null = null;
	let itemImages: string[] = [];
	let stars: number | null = null;
	let cost: number | null = null;
	let chosen: boolean | string = false;

	let costBorder = getCostBorder(cost);
	let starColors = getStars(stars);
	let chosenImage = getChosenImage(chosen);

	function updateImage(champion: Champion | null) {
		if (!champion) {
			championImage = null;
			itemImages = [];
			stars = null;
			cost = null;
			chosen = false;
		} else {
			championImage = getChampionIconImage(champion.name);
			itemImages = champion.items.map((item) => getItemImage(item.name));
			stars = champion.stars;
			cost = champion.cost;
			chosen = champion.chosen;
		}
		costBorder = getCostBorder(cost);
		starColors = getStars(stars);
		chosenImage = getChosenImage(chosen);
	}

	function getCostBorder(cost) {
		let b = `border-["${getCostColor(cost)}"]`;
		console.log(b, cost);
		return b;
	}

	// Call the function whenever champion changes
	$: updateImage(champion);
</script>

<div class="wrapper">
	<div class="star-wrapper">
		{#each starColors as color, index}
			<div class="star inline-block">
				<svg viewBox="0 0 18 18" fill={color}>
					<path
						d="M9.5 14.25l-5.584 2.936 1.066-6.218L.465 6.564l6.243-.907L9.5 0l2.792 5.657 6.243.907-4.517 4.404 1.066 6.218"
					/>
				</svg>
			</div>
		{/each}
	</div>
	{#if champion}
		<div
			class="champion border-4 {costBorder}"
			style="background-image: url({championImage})"
		></div>
		<div class="champion-name text-xs font-sans text-center">
			<p>{champion.name}</p>
		</div>
	{:else}
		<div class="champion border-4 {defaultBorder} {defaultColor}"></div>
	{/if}
	<div class="item-wrapper">
		{#each itemImages as itemImage, index}
			<div class="item inline-block" style="background-image: url({itemImage})" />
		{/each}
	</div>
	{#if chosen}
		<div class="chosen" style="background-image: url({chosenImage})" />
	{:else}
		<div class="chosen" />
	{/if}
</div>

<style>
	.hex-row {
		--m: 4px;
		--w: 100px;
		--h: calc(var(--w) * 0.866);
		--s: calc(var(--w) / 2);
		--tx: calc(calc(var(--h) / 2) + var(--m));
		--ty: calc(calc(var(--w) - var(--s)) / -2);

		position: sticky;
	}

	.champion {
		--size: 60px;
		width: var(--size);
		height: var(--size);
		background-size: var(--size);
		background-position: center;
	}

	.item-wrapper {
		z-index: 2;
		position: absolute;
		bottom: 0px;
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
		height: 33px;
	}

	.item {
		width: 33px;
		height: 33px;
		background-size: 33px;
	}

	.star-wrapper {
		z-index: 2;
		position: absolute;
		top: 1%;
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 2px;
	}

	.star {
		/* height: 100%; */
		width: 18px;
		height: 18px;
	}

	.chosen {
		--cw: calc(70 / 3);
		--ch: calc(82 / 3);
		width: calc(var(--cw) * 1px);
		height: calc(var(--ch) * 1px);
		z-index: 2;
		position: absolute;
		background-position: center;
		background-size: calc(var(--cw) * 1px);
		left: 0px;
		top: 50%;
		transform: translateY(-50%);
	}

	.wrapper {
		position: relative;
		display: inline-flex;
		width: max-content;
	}

	.champion-name {
		--w: 100px;
		z-index: 2;
		position: absolute;
		bottom: calc(var(--w) / 3);
		width: 100%;
		color: white;
		text-shadow:
			-1px -1px 0 #000,
			1px -1px 0 #000,
			-1px 1px 0 #000,
			1px 1px 0 #000;
	}
</style>
