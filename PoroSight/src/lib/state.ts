import { writable } from "svelte/store";

import type { Game } from "$lib/game";

export const gameState = writable<GameState>()

export const game = writable<Game>()

export const currentState = writable<PlayerState>()
export const currentDiff = writable<PlayerDiff>()
export const currentSummary = writable<Summary>()