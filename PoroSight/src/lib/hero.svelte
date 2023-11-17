<script lang="ts">
	import Dropzone from '$lib/dropzone.svelte';
	import GithubLogo from '$lib/navbar/githubLogo.svelte';
	import ThemeChanger from '$lib/navbar/themeChanger.svelte';
	import { poro } from '$lib/image';

	export let gameState: GameState | null = null;
</script>

<div class="hero min-h-screen bg-base-200">
	<div class="hero-content text-center">
		<div class="max-w-md flex flex-col items-center justify-center gap-3">
			<div class="flex flex-col items-center justify-center">
				<img src={poro} class="w-[7rem] h-[7rem] md:w-40 md:h-40" alt="PoroSight Logo" />
				<h1 class="text-4xl md:text-5xl font-bold">
					<span class="text-base-content">Poro</span><span class="text-primary">Sight</span>
				</h1>
			</div>
			<Dropzone bind:gameState />
			<div class="flex flex-row gap-2">
				<GithubLogo />
				<ThemeChanger />
			</div>
		</div>
	</div>
</div>
