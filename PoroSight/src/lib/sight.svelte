<script lang="ts">
	import { gameState } from '$lib/state';
	import { Game } from '$lib/game';

	import Board from '$lib/tft/board.svelte';
	import Navbar from '$lib/navbar/navbar.svelte';
	import Bench from '$lib/tft/bench.svelte';
	import ItemBench from '$lib/tft/itemBench.svelte';
	import Logs from '$lib/tft/logs/logs.svelte';
	import Shop from '$lib/tft/shop.svelte';
	import Scalars from '$lib/tft/scalars.svelte';

	let game = new Game($gameState);
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
	<div>
		<label>
			<input type="number" bind:value={currentPlayerID} min="0" max={players.length - 1} />
			<input type="range" bind:value={currentPlayerID} min="0" max={players.length - 1} />
		</label>
		<label>
			<input type="number" bind:value={currentIndex} min="0" max={maxIndex} />
			<input type="range" bind:value={currentIndex} min="0" max={maxIndex} />
		</label>
	</div>
</div>

<p>playerID: {currentPlayerID} index: {currentIndex} placement: {currentSummary.placement}</p>

<Logs state={currentState} diff={currentDiff} />
