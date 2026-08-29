import { base } from '$app/paths';

const poro = `${base}/images/poro.avif`;

function getChampionShopImage(championName: string) {
    return `${base}/images/champions/shop/${championName}.jpg`;
}

function getChampionIconImage(championName: string) {
    return `${base}/images/champions/icon/${championName}.jpg`;
}

function getItemImage(itemName: string) {
    return `${base}/images/items/${itemName}.png`;
}

function getTraitImage(traitName: string) {
    return `${base}/images/traits/${traitName}.png`;
}

export { poro, getChampionShopImage, getChampionIconImage, getItemImage, getTraitImage };