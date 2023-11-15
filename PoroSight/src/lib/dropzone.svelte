<script>
	import { Game } from '$lib/gamestate';
	let files;

	const processJSON = async (file) => {
		const text = await file.text();
		const game = new Game(text);
	};

	$: if (files) {
		// Note that `files` is of type `FileList`, not an Array:
		// https://developer.mozilla.org/en-US/docs/Web/API/FileList
		console.log(files);

		for (const file of files) {
			console.log(`${file.name}: ${file.size} bytes`);
		}

		processJSON(files[0]);
	}
</script>

<input accept="application/JSON" bind:files id="dropzone-file" name="dropzone-file" type="file" />
