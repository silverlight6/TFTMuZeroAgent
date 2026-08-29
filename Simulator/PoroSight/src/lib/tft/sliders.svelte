<script>
	export let currentPlayerID;
	export let currentIndex;
	export let players;
	export let maxIndex;
</script>

<div class="flex flex-row gap-10">
	<label>
		<p>Current Player (placement):</p>
		<input type="number" bind:value={currentPlayerID} min="0" max={players.length - 1} />
		<input type="range" min="0" max={players.length - 1} bind:value={currentPlayerID} />
	</label>
	<label>
		<p>Action #:</p>
		<input type="number" bind:value={currentIndex} min="0" max={maxIndex} />
		<input type="range" min="0" max={maxIndex} bind:value={currentIndex} />
	</label>
</div>
