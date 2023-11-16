<script>
	import Hero from '$lib/hero.svelte';
	import Sight from '$lib/sight.svelte';
	import { gameState } from '$lib/state';
</script>

{#if $gameState}
	<Sight />
{:else}
	<Hero />
{/if}
