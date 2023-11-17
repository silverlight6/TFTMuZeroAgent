<script lang="ts">
	export let state: PlayerState;
	export let summary: Summary;

	$: streak = state.win_streak > 0 || state.loss_streak > 0;
</script>

<div class="stats bg-base-200 shadow">
	<div class="stat place-items-center">
		<div class="stat-title text-sm">Gold</div>
		<div class="stat-value text-sm">{state.gold}</div>
	</div>

	<div class="stat place-items-center">
		<div class="stat-title text-sm">Health</div>
		<div class="stat-value text-sm text-secondary">{state.health}</div>
		<div class="stat-desc">placement: {summary.placement}</div>
	</div>

	<div class="stat place-items-center">
		<div class="stat-title text-sm">Level</div>
		<div class="stat-value text-sm">{state.level}</div>
		<div class="stat-desc">exp: {state.exp}</div>
	</div>

	{#if streak}
		<div class="stat place-items-center">
			{#if state.win_streak > 0}
				<div class="stat-title text-sm text-orange-600">Win Streak</div>
				<div class="stat-value text-sm">{state.win_streak}</div>
			{/if}
			{#if state.loss_streak > 0}
				<div class="stat-title text-sm text-blue-500">Loss Streak</div>
				<div class="stat-value text-sm">{state.loss_streak}</div>
			{/if}
		</div>
	{/if}
</div>
