<script>
	import GithubLogo from './githubLogo.svelte';
	import ThemeChanger from './themeChanger.svelte';
	import { poro } from '$lib/image';
</script>

<div class="navbar bg-base-100">
	<div class="flex-1">
		<a class="btn btn-ghost text-xl">
			<img src={poro} class="w-7 h-7 md:w-10 md:h-10" alt="PoroSight Logo" />
			<p class="font-title text-lg md:text-2xl">
				<span class="text-base-content">Poro</span><span class="text-primary">Sight</span>
			</p>
		</a>
	</div>
	<div class="flex-none">
		<GithubLogo />
	</div>
	<div class="flex-none m-5">
		<ThemeChanger />
	</div>
</div>
