<script lang="ts">
	import { Game } from '$lib/game';

	import Board from '$lib/tft/board.svelte';
	import Navbar from '$lib/navbar/navbar.svelte';
	import Bench from '$lib/tft/bench.svelte';
	import ItemBench from '$lib/tft/itemBench.svelte';
	import Logs from '$lib/tft/logs/logs.svelte';
	import Shop from '$lib/tft/shop.svelte';
	import Scalars from '$lib/tft/scalars.svelte';
	import ScrollableSteps from './scrollableSteps.svelte';

	export let gameState: GameState;

	let game = new Game(gameState);
	let currentPlayerID = 0;
	let currentIndex = 0;

	$: kind = game.getKind();
	$: showEconomy = kind === 'full_game' || kind === 'single_player';
	$: showShop = showEconomy;
	$: showOpponentBoard = kind === 'position' || kind === 'item';
	$: showPlayerSlider = kind === 'full_game';
	$: kindLabel = {
		full_game: 'Full game',
		single_player: 'Single player',
		position: 'Position',
		item: 'Item'
	}[kind];

	// Players
	$: players = game.getPlayers();
	$: currentPlayer = players[currentPlayerID] ?? players[0];
	$: opponentID = players.find((id) => id !== currentPlayer);

	// State
	$: [currentState, currentDiff] = game.getPlayerState(currentPlayer, currentIndex);
	$: currentSummary = game.getPlayerSummary(currentPlayer);
	$: currentTimeSteps = game.getPlayerSteps(currentPlayer);
	$: opponentState = opponentID ? game.getPlayerState(opponentID, 0)[0] : null;
	$: latestBattle = currentTimeSteps[currentIndex]?.battle
		?? currentTimeSteps.filter((step) => step.battle).slice(-1)[0]?.battle;
	$: opponentBoard = latestBattle?.opponentBoard ?? opponentState?.board ?? [];
</script>

<Navbar />

<div class="flex flex-col items-center flex-initial">
	<p class="text-sm opacity-70 mb-1">{kindLabel}</p>
	<ScrollableSteps
		timeSteps={currentTimeSteps}
		bind:currentIndex
		bind:currentPlayerID
		playerIds={players}
		showPlayerSlider={showPlayerSlider}
	/>
	<Board board={currentState.board} />
	{#if showOpponentBoard}
		<p class="text-xs opacity-70 mt-2">{opponentID ?? latestBattle?.opponent ?? 'opponent'}</p>
		<Board board={opponentBoard} />
	{/if}
	<ItemBench items={currentState.items} />
	<Bench bench={currentState.bench} />
	{#if showEconomy}
		<Scalars state={currentState} summary={currentSummary} />
	{/if}
	{#if showShop}
		<Shop shop={currentState.shop} />
	{/if}
</div>

{#if kind === 'full_game'}
	<p>playerID: {currentPlayerID} index: {currentIndex} placement: {currentSummary.placement}</p>
{:else}
	<p>player: {currentPlayer} index: {currentIndex}</p>
{/if}

<Logs state={currentState} diff={currentDiff} />
