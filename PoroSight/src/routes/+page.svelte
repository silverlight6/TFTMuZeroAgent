<script lang="ts">
	import Hero from '$lib/hero.svelte';
	import Sight from '$lib/sight.svelte';

	let gameState: GameState | null = null;
</script>

{#if gameState}
	<Sight {gameState} />
{:else}
	<Hero bind:gameState />
{/if}
