<script>
	import { gameState } from '$lib/state';
	import { GameState } from '$lib/schema';
	import { writable } from 'svelte/store';

	let files = null;
	let file = null;

	let error = false;

	const fileError = (file) => {
		console.log('Invalid File');
		error = true;
	};

	const fileSuccess = (file) => {
		console.log('Valid File');
		error = false;
	};

	$: if (files) {
		const fileText = files[0].text();
		fileText.then((text) => {
			file = text;
		});
		files = null;
	}

	$: if (file) {
		try {
			const game = GameState.parse(JSON.parse(file));
			gameState.set(game);
			console.log($gameState);
			fileSuccess(file);
		} catch (e) {
			console.log('File Parse Error');
			console.log(e);
			fileError(file);
		}
		file = null;
	}
</script>

<!-- <div class="form-control w-full max-w-xs">
	{#if !error}
		<label class="label">
			<span class="label-text">{label}</span>
		</label>
	{:else}
		<label class="label">
			<span class="label-text text-error">{errorLabel}</span>
		</label>
	{/if}
	<input
		accept="application/json"
		bind:files
		type="file"
		class="file-input file-input-bordered {theme} w-full max-w-xs"
	/>
</div> -->

<div class="flex items-center justify-center w-full">
	<label
		for="dropzone-file"
		class="flex flex-col items-center justify-center w-[20rem] h-32 border-2 border-dashed rounded-lg cursor-pointer
        "
	>
		<div class="flex flex-col items-center justify-center pt-5 pb-6">
			<svg
				class="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
				aria-hidden="true"
				xmlns="http://www.w3.org/2000/svg"
				fill="none"
				viewBox="0 0 20 16"
			>
				<path
					stroke="currentColor"
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
				/>
			</svg>
			<p class="mb-2 text-sm text-gray-500 dark:text-gray-400">
				<span class="font-semibold">Click to upload</span> or drag and drop
			</p>
			<!-- <p class="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG or GIF (MAX. 800x400px)</p> -->
		</div>
		<input accept="application/json" bind:files id="dropzone-file" type="file" class="hidden" />
	</label>
</div>
