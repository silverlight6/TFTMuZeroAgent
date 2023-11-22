import { getChampionIconImage, getChampionShopImage, getItemImage, getTraitImage } from "$lib/image";

// Config variables
export let colors = {
    oneCost: '#afafaf',
    twoCost: '#1bc660',
    threeCost: '#0b6cc3',
    fourCost: '#f947c6',
    fiveCost: '#fe8900',

    oneStar: '#cd7f32',
    twoStar: '#C0C0C0',
    threeStar: '#FFD700',

    defaultBg: '#1e1e1e',
    defaultBorder: '#1e1e1e',

    chosenColor: '#d186e1'
}

function getIconImage(champion: Champion) {
    return getChampionIconImage(champion.name);
}

function getShopImage(champion: Champion) {
    return getChampionShopImage(champion.name);
}

function getItemImages(items: Item[]) {
    return items.map(item => getItemImage(item.name));
}

function getChosenImage(champion: Champion) {
    return getTraitImage(champion.chosen);
}

function getCostColor(champion: Champion) {
    switch (champion.cost) {
        case 1:
            return colors.oneCost;
        case 2:
            return colors.twoCost;
        case 3:
            return colors.threeCost;
        case 4:
            return colors.fourCost;
        case 5:
            return colors.fiveCost;
        default:
            return colors.defaultBorder;
    }
}


function getStarColor(stars: number) {
    switch (stars) {
        case 1:
            return colors.oneStar;
        case 2:
            return colors.twoStar;
        case 3:
            return colors.threeStar;
        default:
            return colors.threeStar;
    }
}

export let utils = { getIconImage, getShopImage, getItemImages, getChosenImage, getCostColor, getStarColor }
