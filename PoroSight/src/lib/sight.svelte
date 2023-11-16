<script>
	import { Game } from '$lib/game';
	import { gameState, game, currentState, currentDiff } from './state';
	import Board from '$lib/tft/board.svelte';
	import Navbar from '$lib/navbar/navbar.svelte';
	import Bench from './tft/bench.svelte';
	import ItemBench from './tft/itemBench.svelte';
	import Logs from './tft/logs.svelte';
	import Shop from './tft/shop.svelte';
	import Scalars from './tft/scalars.svelte';

	$game = new Game($gameState);

	const players = $game.getPlayers();

	let currentPlayerID = 0;
	let currentPlayer = players[currentPlayerID];

	let maxIndex = $game.getPlayerLength(currentPlayer) - 1;

	let index = 0;

	[$currentState, $currentDiff] = $game.getPlayerState(currentPlayer, index);

	$: {
		[$currentState, $currentDiff] = $game.getPlayerState(currentPlayer, index);
	}

	$: {
		maxIndex = $game.getPlayerLength(currentPlayer);
		index = Math.min(index, maxIndex);
	}

	$: {
		currentPlayer = players[currentPlayerID];
	}
</script>

<Navbar />

<Board />
<ItemBench />
<Bench />

<Scalars />
<Shop />

<label>
	<input type="number" bind:value={currentPlayerID} min="0" max={players.length - 1} />
	<input type="range" bind:value={currentPlayerID} min="0" max={players.length - 1} />
</label>

<label>
	<input type="number" bind:value={index} min="0" max={maxIndex} />
	<input type="range" bind:value={index} min="0" max={maxIndex} />
</label>

<p>{currentPlayer} {currentPlayerID} {index}</p>

<Logs />
