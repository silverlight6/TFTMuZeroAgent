<script lang="ts">
	import { createLog } from './logs';

	export let state: PlayerState;
	export let diff: PlayerDiff;

	$: log = createLog(state, diff);
</script>

{#each log as line}
	<p class="text-sm text-base-content">{line}</p>
{/each}
