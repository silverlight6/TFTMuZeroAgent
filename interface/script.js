//Don't look in here. Please.
//I hate writing UI stuff so just beat up the keyboard until got some working version done.
//The code is ugly as shit.

champions = [
    'aatrox',
    'ahri',
    'akali',
    'annie',
    'aphelios',
    'ashe',
    'azir',
    'cassiopeia',
    'diana',
    'elise',
    'evelynn',
    'ezreal',
    'fiora',
    'garen', 
    'hecarim',
    'irelia',
    'janna',
    'jarvaniv',
    'jax',
    'jhin',
    'jinx',
    'kalista',
    'katarina',
    'kayn',
    'kennen',
    'kindred',
    'leesin',
    'lillia',
    'lissandra',
    'lulu',
    'lux',
    'maokai',
    'morgana',
    'nami',
    'nidalee',
    'nunu',
    'pyke',
    'riven',
    'sejuani',
    'sett',
    'shen',
    'sylas',
    'tahmkench',
    'talon',
    'teemo',
    'thresh',
    'twistedfate',
    'vayne', 
    'veigar',
    'vi',
    'warwick',
    'wukong',
    'xinzhao',
    'yasuo',
    'yone',
    'yuumi',
    'zed',
    'zilean',
    'construct',
    'galio',
    'sandguard',
]


items = [
    'bf_sword',
    'chain_vest',
    'giants_belt',
    'needlessly_large_rod',
    'negatron_cloak',
    'recurve_bow',
    'sparring_gloves', 
    'spatula', 
    'tear_of_the_goddess',
    'bloodthirster',
    'blue_buff',
    'bramble_vest',
    'chalice_of_power',
    'deathblade',
    'dragons_claw',
    'duelists_zeal',
    'elderwood_heirloom',
    'force_of_nature',
    'frozen_heart',
    'gargoyle_stoneplate',    
    'giant_slayer',
    'guardian_angel',  
    'guinsoos_rageblade',
    'hand_of_justice',  
    'hextech_gunblade',
    'infinity_edge',
    'ionic_spark',
    'jeweled_gauntlet',
    'last_whisper',
    'locket_of_the_iron_solari',
    'ludens_echo',
    'mages_cap',            
    'mantle_of_dusk',           
    'morellonomicon',
    'quicksilver',
    'rabadons_deathcap',
    'rapid_firecannon',
    'redemption',
    'runaans_hurricane',
    'shroud_of_stillness',
    'spear_of_shojin',
    'statikk_shiv',
    'sunfire_cape',
    'sword_of_the_divine',
    'thiefs_gloves',
    'titans_resolve',
    'trap_claw',
    'vanguards_cuirass',
    'warlords_banner',
    'warmogs_armor',
    'youmuus_ghostblade',
    'zekes_herald',
    'zephyr',
    'zzrot_portal'
]


let body = document.getElementsByTagName('body')[0]

        for (let i = 0; i < 8; i++){
            let line = document.createElement('div')
            line.classList.add('line')
            for(let j = 0; j < 7; j++){
                
                let hex = document.createElement('div')
                hex.style.float ="left"

                y = 8 - i - 1
                x = j.toString()
                y = y.toString()
                hex.id = y+x
                
                hex.classList.add('hex')
                hex.style.height = "100px"
                hex.style.width = "100px"
                hex.style.margin = "2px"

                hex.style.border = "1px solid black"

                //champions
                select = document.createElement('select')
                select.classList.add('champion')
                for(let c = 0; c < champions.length+1; c++){
                    let option = document.createElement('option')

                    if(c == 0){
                        option.value = ""
                        option.innerHTML = ""
                    }
                    else{
                        option.value = champions[c-1]
                        option.innerHTML = champions[c-1]
                    }
                    select.append(option)
                }
                hex.append(select)

                //stars
                select = document.createElement('select')
                select.classList.add('stars')
                for(let c = 0; c < 4; c++){

                    let option = document.createElement('option')
                    option.value = c + 1
                    option.innerHTML = c + 1
                    select.append(option)
                    
                }
                hex.append(select)
                
                //item0
                select = document.createElement('select')
                select.classList.add("item0")
                    for(let c = 0; c < items.length + 1; c++){
                    let option = document.createElement('option')

                    if(c == 0){
                        option.value = ""
                        option.innerHTML = ""
                    }
                    else{
                        option.value = items[c-1]
                        option.innerHTML = items[c-1]
                    }
                    select.style.width = "20px"
                    select.append(option)
                }
                hex.append(select)

                //item1
                select = document.createElement('select')
                select.classList.add("item1")
                    for(let c = 0; c < items.length + 1; c++){
                    let option = document.createElement('option')

                    if(c == 0){
                        option.value = ""
                        option.innerHTML = ""
                    }
                    else{
                        option.value = items[c-1]
                        option.innerHTML = items[c-1]
                    }
                    select.style.width = "20px"
                    select.append(option)
                }
                hex.append(select)

                //item2
                select = document.createElement('select')
                select.classList.add("item2")
                    for(let c = 0; c < items.length + 1; c++){
                    let option = document.createElement('option')

                    if(c == 0){
                        option.value = ""
                        option.innerHTML = ""
                    }
                    else{
                        option.value = items[c-1]
                        option.innerHTML = items[c-1]
                    }
                    select.style.width = "20px"
                    select.append(option)
                }
                hex.append(select)
                
                daddy = document.createElement('input')
                daddy.placeholder = 'azir'
                daddy.classList.add('daddy_coordinates')
                daddy.style.width = "40px"
                hex.append(daddy)

                chosen = document.createElement('input')
                chosen.placeholder = 'chosen'
                chosen.classList.add('chosen')
                chosen.style.width = "43px"
                hex.append(chosen)

                line.append(hex)

                if(i % 2 == 1 && j == 0){
                    hex.style.marginLeft = "50px"
                } 
            }
                line.style.float ="left"

            body.append(line)
        }
        
        body.style.width = "820px"

        button = document.createElement('button')
        button.id = "submit"
        button.innerHTML = "Generate data"
        button.style.backgroundColor = '#66ff66'
        button.style.height = '40px'
        button.style.width = '120px'
        button.style.border = '2px solid #00cc00'
        
        body.append(button)

        div = document.createElement('div')

        box = document.createElement('textarea')
        box.style.width = '600px'
        box.style.height = '300px'
        div.append(box)
        body.append(div)


        blue = document.createElement('div')
        blue.style.height = '430px'
        blue.style.width = '10px'
        blue.style.position = 'absolute'
        blue.style.top = '425px'
        blue.style.left = '0px'
        blue.style.backgroundColor = 'rgb(3, 136, 252)'
        body.append(blue)

        red = document.createElement('div')
        red.style.height = '430px'
        red.style.width = '10px'
        red.style.position = 'absolute'
        red.style.top = '0px'
        red.style.left = '0px'
        red.style.backgroundColor = 'rgb(255, 107, 129)'
        body.append(red)

        button.addEventListener('click', () => {
            json = {}
            json['blue'] = []
            json['red'] = []

            for(let i = 0; i < 11; i++){
                line = body.children[i]
                if(line.classList.contains('line')){
                    for(let j = 0; j < 7; j++){
                        let hex = line.children[j]
                        hex_object = {}
                        if(hex.children[0].value){
                            hex_object['name'] = hex.children[0].value
                            hex_object['stars'] = hex.children[1].value
                            hex_object['items'] = []
                            if(hex.children[2].value.length > 1){hex_object['items'].push(hex.children[2].value)}
                            if(hex.children[3].value.length > 1){hex_object['items'].push(hex.children[3].value)}
                            if(hex.children[4].value.length > 1){hex_object['items'].push(hex.children[4].value)}
                            hex_object["overlord_coordinates"] = false
                            if(hex.children[5].value.length > 1) {
                                hex_object['overlord_coordinates'] = [hex.children[5].value.split(" ")[0], hex.children[5].value.split(" ")[1]]
                            }
                            hex_object["chosen"] = false
                            if(hex.children[6].value.length > 1) {hex_object['chosen'] = hex.children[6].value}
                            hex_object["y"] = hex.id[0]
                            hex_object["x"] = hex.id[1]

                            if(hex.id[0] < 4) json['blue'].push(hex_object)
                            if(hex.id[0] >= 4) json['red'].push(hex_object)
                        }

                    }
                }
            }
            box.value = JSON.stringify(json)
        })
