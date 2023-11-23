<script lang="ts">
	import { utils } from '$lib/tft/champion/utils';

	export let stars: number;

	$: starColors = createStarColors(stars);

	function createStarColors(stars: number) {
		const color = utils.getStarColor(stars);
		return Array(stars).fill(color);
	}
</script>

{#each starColors as color, _}
	<div class="star inline-block">
		<svg viewBox="0 0 18 18" fill={color}>
			<path
				d="M9.5 14.25l-5.584 2.936 1.066-6.218L.465 6.564l6.243-.907L9.5 0l2.792 5.657 6.243.907-4.517 4.404 1.066 6.218"
			/>
		</svg>
	</div>
{/each}

<style>
	.star {
		width: var(--star-size);
		height: var(--star-size);
	}
</style>
