<script lang="ts">
	import { getChampionIconImage, getItemImage, getChosenImage } from '$lib/image';
	import { getCostBorder, getStars } from '$lib/util';

	export let champion: Champion | null | undefined = null;

	let defaultHexColor: string = 'bg-base-content';
	let defaultHexBorder: string = 'bg-primary';

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
		<div class="hexagon-wrapper {costBorder}">
			<div class="hexagon" style="background-image: url({championImage})" />
		</div>
		<div class="champion-name text-xs font-sans text-center">
			<p>{champion.name}</p>
		</div>
	{:else}
		<div class="hexagon-wrapper {defaultHexBorder}">
			<div class="hexagon {defaultHexColor}"></div>
		</div>
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
	.hexagon-wrapper {
		width: var(--h);
		height: var(--w);
		margin: 6px;

		display: inline-block;
		font-size: initial;
		clip-path: polygon(0% 25%, 0% 75%, 50% 100%, 100% 75%, 100% 25%, 50% 0%);
	}

	.hexagon {
		width: calc(var(--h) * 0.9);
		height: calc(var(--w) * 0.9);
		margin: 0 auto;

		display: inline-block;
		font-size: initial;
		clip-path: polygon(0% 25%, 0% 75%, 50% 100%, 100% 75%, 100% 25%, 50% 0%);
		background-size: calc(var(--w) * 0.8);
		background-position: center;

		position: absolute;
		left: 50%;
		top: 50%;
		transform: translate(-50%, -50%);
	}

	.item-wrapper {
		z-index: 2;
		position: absolute;
		bottom: 0px;
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
		height: var(--ih);
	}

	.item {
		--iw: calc(var(--w) / 3);
		--ih: var(--iw);
		width: var(--iw);
		height: var(--ih);
		background-size: var(--iw);
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
