# Simulator docs

One page per environment. Each env has its own action space, observation, and reward, so use the page that matches the env you are wiring up.

| Env | Doc |
|-----|-----|
| Full 8-player game (`parallel_env` / `env`) | [full_game.md](full_game.md) |
| Positioning (`TFT_Position_Simulator`) | [position.md](position.md) |
| Item assignment (`TFT_Item_Simulator`) | [item.md](item.md) |
| Solo campaign (`TFT_Single_Player_Simulator`) | [single_player.md](single_player.md) |
| Batched position (`TFT_Vector_Pos_Simulator`) | [vector_position.md](vector_position.md) |
| Batched solo campaign (`TFT_Single_Player_Vector_Simulator`) | [vector_single_player.md](vector_single_player.md) |

Runnable loops live under `examples/`.
