<script type="ts">
	import { currentState, currentDiff } from '$lib/state';
</script>

<div class="stats shadow">
	<div class="stat place-items-center">
		<div class="stat-title text-sm">Gold</div>
		<div class="stat-value text-sm">{$currentState.gold}</div>
	</div>

	<div class="stat place-items-center">
		<div class="stat-title text-sm">Health</div>
		<div class="stat-value text-sm text-secondary">{$currentState.health}</div>
	</div>

	<div class="stat place-items-center">
		<div class="stat-title text-sm">Level</div>
		<div class="stat-value text-sm">{$currentState.level}</div>
		<div class="stat-desc">exp: {$currentState.exp}</div>
	</div>

	<div class="stat place-items-center">
		{#if $currentState.win_streak > 0}
			<div class="stat-title text-sm text-orange-600">Win Streak</div>
			<div class="stat-value text-sm">{$currentState.win_streak}</div>
		{/if}
		{#if $currentState.loss_streak > 0}
			<div class="stat-title text-sm text-blue-500">Loss Streak</div>
			<div class="stat-value text-sm">{$currentState.loss_streak}</div>
		{/if}
	</div>
</div>
