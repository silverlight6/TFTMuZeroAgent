<script lang="ts">
	import { Game } from '$lib/game';

	import Board from '$lib/tft/board.svelte';
	import Navbar from '$lib/navbar/navbar.svelte';
	import Bench from '$lib/tft/bench.svelte';
	import ItemBench from '$lib/tft/itemBench.svelte';
	import Logs from '$lib/tft/logs/logs.svelte';
	import Shop from '$lib/tft/shop.svelte';
	import Scalars from '$lib/tft/scalars.svelte';
	import Sliders from './tft/sliders.svelte';

	export let gameState: GameState;

	let game = new Game(gameState);
	let currentPlayerID = 0;
	let currentIndex = 0;

	// Players
	$: players = game.getPlayers();
	$: currentPlayer = players[currentPlayerID];

	// Index
	$: maxIndex = game.getPlayerLength(currentPlayer);
	$: {
		currentIndex = Math.min(currentIndex, maxIndex);
	}

	// State
	$: [currentState, currentDiff] = game.getPlayerState(currentPlayer, currentIndex);
	$: currentSummary = game.getPlayerSummary(currentPlayer);
</script>

<Navbar />

<div class="flex flex-col items-center flex-initial">
	<Board board={currentState.board} />
	<ItemBench items={currentState.items} />
	<Bench bench={currentState.bench} />
	<Scalars state={currentState} summary={currentSummary} />
	<Shop shop={currentState.shop} />
	<div class="justify-start">
		<Sliders bind:currentPlayerID bind:currentIndex {players} {maxIndex} />
	</div>
</div>

<p>playerID: {currentPlayerID} index: {currentIndex} placement: {currentSummary.placement}</p>

<Logs state={currentState} diff={currentDiff} />
