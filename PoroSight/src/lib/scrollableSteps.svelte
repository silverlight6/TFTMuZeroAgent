<script lang="ts">
	import { onMount } from 'svelte';
	import Battle from './tft/battle.svelte';
	import { computeRoundString } from './tft/util';

	export let timeSteps: TimeStep[];
	export let currentIndex: number;
	export let currentPlayerID: number;

	let gap = 1;
	let battleWidth = 32 + gap;
	let actionWidth = 20 + gap;

	let lock = false; // scroll lock

	let tick = 750; // Default tick is 1s
	let speed = 1; // Default speed is 1x
	let intervalID;

	$: direction = speed > 0 ? 1 : -1;

	function atEnd() {
		return (
			(currentIndex === 0 && direction === -1) ||
			(currentIndex === timeSteps.length - 1 && direction === 1)
		);
	}

	function play() {
		if (atEnd()) {
			return;
		}
		lock = true;
		currentIndex += direction;
		scrollToActive(currentIndex);

		intervalID = setInterval(
			() => {
				if (atEnd()) {
					pause();
					return;
				}
				currentIndex += direction;
				scrollToActive(currentIndex);
			},
			tick / Math.abs(speed)
		);
	}

	function pause() {
		lock = false;
		if (intervalID !== undefined) {
			clearInterval(intervalID);
		}
	}

	const calcActionsPerRound = (timeSteps: TimeStep[]) => {
		let actions = 0;
		while (!timeSteps[actions + 1].battle) {
			actions += 1;
		}
		actions += 1; // First round has 1 less action the all others
		let roundSize = battleWidth + actionWidth * actions;
		let firstRoundSize = roundSize - actionWidth;
		return [actions, roundSize, firstRoundSize];
	};

	$: [actionsPerRound, roundSize, firstRoundSize] = calcActionsPerRound(timeSteps);
	$: Math.min(currentIndex, timeSteps.length - 1);

	let leftArrow;
	let rightArrow;
	let doubleLeftArrow;
	let doubleRightArrow;

	let roundsList;

	function setIndex(index) {
		currentIndex = index;
		scrollToActive(currentIndex);
	}

	function toStart() {
		currentIndex = 0;
		scrollToActive(currentIndex);
	}

	function toEnd() {
		currentIndex = timeSteps.length - 1;
		scrollToActive(currentIndex);
	}

	function findPrevBattle(index: number) {
		if (index === 0) {
			return index;
		}

		let prevBattle = index - 1;

		while (!timeSteps[prevBattle].battle) {
			prevBattle -= 1;
		}

		return prevBattle;
	}

	function findNextBattle(index: number) {
		if (index === timeSteps.length - 1) {
			return index;
		}

		let nextBattle = index + 1;

		while (!timeSteps[nextBattle].battle) {
			nextBattle += 1;
		}

		return nextBattle;
	}

	function leftArrowClick() {
		if (currentIndex === 0) {
			return;
		}

		currentIndex = currentIndex - 1;
		scrollToActive(currentIndex);
	}

	function rightArrowClick() {
		if (currentIndex === timeSteps.length - 1) {
			return;
		}

		currentIndex = currentIndex + 1;
		scrollToActive(currentIndex);
	}

	function doubleLeftArrowClick() {
		currentIndex = findPrevBattle(currentIndex);
		scrollToActive(currentIndex);
	}

	function doubleRightArrowClick() {
		currentIndex = findNextBattle(currentIndex);
		scrollToActive(currentIndex);
	}

	function isActive(index, currentIndex, isBattle = false) {
		if (isBattle) {
			return index === currentIndex
				? transitionBattle
				: 'bg-secondary-content text-neutral-content';
		}
		return index === currentIndex ? transitionAction : 'bg-secondary-content';
	}

	const transitionAction = 'active transition-all -translate-y-1 duration-50 bg-secondary';
	const transitionBattle = 'active transition-all duration-100 bg-secondary text-secondary-content';
	let btn = 'btn btn-xs sm:btn-sm btn-outline';

	function battleOutcome(battle) {
		if (battle.health < 0) {
			return 'death';
		} else if (battle.result === 'win') {
			return 'win';
		} else {
			return 'lose';
		}
	}

	function isCarousel(battle) {
		if (battle.round < 6) return false;
		return (battle.round - 6) % 6 === 0;
	}

	function getBattleIcon(battle) {
		if (battle.opponent === 'minion') {
			// return '🐺';
			return 'M';
		} else if (isCarousel(battle)) {
			// return '🎠';
			return 'C';
		} else {
			return 'B';
		}
	}

	function showModal(battle) {
		const modal = document.getElementById(`battle_${battle.round}`);
		modal.showModal();
	}

	function scrollToActive(currentIndex) {
		let index = currentIndex;
		let scrollLeft = roundsList.scrollLeft;
		let round;

		if (currentIndex === timeSteps.length - 1) {
			// 80vw
			const width = window.innerWidth * 0.8;
			const padding = (width - battleWidth) / 2 - 1.5;
			scrollLeft = roundsList.scrollWidth - battleWidth - padding;
		} else if (currentIndex < actionsPerRound) {
			round = 0;
			scrollLeft = currentIndex === 0 ? 0 : (currentIndex - 1) * actionWidth + battleWidth;
		} else {
			index -= actionsPerRound; // First round
			round = Math.floor(index / (actionsPerRound + 1));
			index -= (actionsPerRound + 1) * round; // All other rounds
			scrollLeft = index === 0 ? 0 : (index - 1) * actionWidth + battleWidth;
			scrollLeft += firstRoundSize + round * roundSize;
		}
		roundsList.scrollTo({ left: scrollLeft });
	}

	onMount(() => {
		// Not my best work, but it works...
		roundsList.addEventListener('scroll', () => {
			if (lock) return;

			let scrollLeft = roundsList.scrollLeft;
			let width = roundsList.clientWidth;
			// Compute the active element using battleWidth and actionWidth
			let round;
			let index;

			if (scrollLeft < firstRoundSize) {
				round = 0;
				scrollLeft -= battleWidth;
				index = scrollLeft < 0 ? 0 : Math.floor(scrollLeft / actionWidth) + 1;
				currentIndex = index;
			} else {
				scrollLeft -= firstRoundSize;
				round = Math.floor(scrollLeft / roundSize);
				scrollLeft -= round * roundSize;
				scrollLeft -= battleWidth; // Don't question it...
				index = scrollLeft < 0 ? 0 : Math.floor(scrollLeft / actionWidth) + 1;

				index += actionsPerRound; // First round
				index += (actionsPerRound + 1) * round; // All other rounds
			}

			if (currentIndex !== index) {
				currentIndex = index;
			}
		});
	});
</script>

<div class="scrollable-container">
	<div class="flex justify-center gap-5 mb-3 align-middle">
		<div class="text-center align-middle text-lg stat-value mb-3">
			Round {computeRoundString(timeSteps[currentIndex].round)}
		</div>
		<div>
			<div class="w-full flex justify-between text-xs px-2">
				<span>1</span>
				<span>2</span>
				<span>3</span>
				<span>4</span>
				<span>5</span>
				<span>6</span>
				<span>7</span>
				<span>8</span>
			</div>
			<input bind:value={currentPlayerID} type="range" min="0" max="7" class="range range-xs" />
		</div>
	</div>
	<div class="time-steps mb-3">
		<ul
			bind:this={roundsList}
			class="rounds-list flex gap-[1px] select-none whitespace-nowrap overflow-x-scroll relative"
		>
			<li>
				<div class="padding" />
			</li>
			{#each timeSteps as step, index}
				{#if step.battle}
					<li>
						<button
							class="btn btn-sm battle rounded-md border-2 w-5 {isActive(
								index,
								currentIndex,
								true
							)} {battleOutcome(step.battle)}"
							on:click={() => {
								setIndex(index);
								showModal(step.battle);
							}}
						>
							{getBattleIcon(step.battle)}
						</button>
						<dialog id="battle_{step.battle.round}" class="modal modal-bottom sm:modal-middle">
							<div class="modal-box">
								<Battle battle={step.battle} {currentPlayerID} />
							</div>
							<form method="dialog" class="modal-backdrop">
								<button>close</button>
							</form>
						</dialog>
					</li>
				{:else}
					<li class="flex align-middle items-center">
						<button
							class="inline-block rounded-sm align-middle w-[20px] h-[10px] m-auto {isActive(
								index,
								currentIndex
							)}"
							on:click={() => setIndex(index)}
						></button>
					</li>
				{/if}
			{/each}
			<li>
				<div class="padding" />
			</li>
		</ul>
		<div class="split-line mx-auto"></div>
	</div>

	<div class="buttons flex justify-center mb-3">
		<div class="left-items">
			<button on:click={toStart} class={btn}> ST </button>

			<button
				bind:this={doubleLeftArrow}
				on:click={doubleLeftArrowClick}
				class="double-left-arrow {btn}"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke="currentColor"
					class="w-6 h-6"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M18.75 19.5l-7.5-7.5 7.5-7.5m-6 15L5.25 12l7.5-7.5"
					/>
				</svg>
			</button>
			<button bind:this={leftArrow} on:click={leftArrowClick} class="left-arrow {btn}">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke="currentColor"
					class="w-6 h-6"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
				</svg>
			</button>
		</div>
		<div class="middle-items">
			{#if lock}
				<button on:click={pause} class={btn}>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 24 24"
						fill="currentColor"
						class="w-6 h-6"
					>
						<path
							fill-rule="evenodd"
							d="M6.75 5.25a.75.75 0 01.75-.75H9a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H7.5a.75.75 0 01-.75-.75V5.25zm7.5 0A.75.75 0 0115 4.5h1.5a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H15a.75.75 0 01-.75-.75V5.25z"
							clip-rule="evenodd"
						/>
					</svg>
				</button>
			{:else}
				<button on:click={play} class={btn}>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 24 24"
						fill="currentColor"
						class="w-6 h-6"
					>
						<path
							fill-rule="evenodd"
							d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z"
							clip-rule="evenodd"
						/>
					</svg>
				</button>
			{/if}
			<select
				on:click={pause}
				bind:value={speed}
				class="ml-2 select select-bordered select-xs w-full max-w-xs sm:select-sm"
			>
				<option value="-4">-4x</option>
				<option value="-2">-2x</option>
				<option value="-1">-1x</option>
				<option value="1" selected>1x</option>
				<option value="2">2x</option>
				<option value="4">4x</option>
			</select>
		</div>
		<div class="right-items">
			<button bind:this={rightArrow} on:click={rightArrowClick} class="right-arrow {btn}">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke="currentColor"
					class="w-6 h-6"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
				</svg>
			</button>

			<button
				bind:this={doubleRightArrow}
				on:click={doubleRightArrowClick}
				class="double-right-arrow {btn}"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke="currentColor"
					class="w-6 h-6"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M11.25 4.5l7.5 7.5-7.5 7.5m-6-15l7.5 7.5-7.5 7.5"
					/>
				</svg>
			</button>
			<button on:click={toEnd} class={btn}> ED </button>
		</div>
	</div>
</div>

<style>
	.scrollable-container ul {
		-ms-overflow-style: none;
		scrollbar-width: none;
	}
	.scrollable-container ul::-webkit-scrollbar {
		display: none;
	}

	.time-steps {
		height: 32px;
		max-width: var(--max-scrollable-width);
	}

	.rounds-list {
		max-width: var(--max-scrollable-width);
	}

	.padding {
		width: calc((var(--max-scrollable-width) / 2) - 1.5px);
		height: 100%;
	}

	.split-line {
		width: 1px;
		height: 32px;
		border: 0.5px dashed white;
		transform: translateY(-100%);
	}

	.battle {
		width: 2rem;
		height: 2rem;
		text-align: center;
	}

	.left-items,
	.middle-items,
	.right-items {
		display: flex;
		height: 100%;
		top: 0;
		align-items: center;
		padding: 0 10px;
	}

	.left-items {
	}

	.right-items {
		right: 0;
		justify-content: flex-end;
	}

	.win {
		border-color: green;
	}

	.lose {
		border-color: red;
	}

	.death {
		border-color: gray;
	}
</style>
