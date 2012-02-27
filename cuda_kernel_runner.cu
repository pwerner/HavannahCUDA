/*	Havannah CUDA – This program provides a CUDA based playout strategy for Monte-Carlo-Evaluation
 *	Copyright (C) 2012  Peter Werner
 *	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *	Contact: Peter@wernerbrothers.de
 */

#include "cuda_kernel_runner.h"
#include <stdio.h>
/* kernel runner that copies always the active game to device and back after running kernel */
void rateGame1(const small_board * b) {
	game * games_d;
	// prepare kernel call
	dim3 dimBlock( 1, 1 );
	dim3 dimGrid( MAX_ACTIVE_GAMES, 1 );
	// init rng
	curandState *rng;
	cudaMalloc((void **)&rng, MAX_ACTIVE_GAMES* sizeof(curandState));
	setup_rnd_kernel<<<dimBlock, dimGrid>>>(rng, time(0));
	cudaThreadSynchronize();
	// get free moves on board
	idx toPlay[ELEMENT_SIZE];
	short lastIndex = init_toplay(toPlay,b);
	int gameCount = lastIndex +1;
	// create a game for each move
	game openGames[ELEMENT_SIZE];
	int i;
	for(i =0; i < gameCount; i++) {
		init_game(&openGames[i],toPlay[i]);
	}
	// copy original board
	small_board *b_d;
	cudaMalloc( (void**)&b_d, sizeof(small_board));
	cudaMemcpy( b_d, b, sizeof(small_board), cudaMemcpyHostToDevice);
	// always MAX_ACTIVE_GAMES active
	for(i = 0; i < gameCount; i=i+MAX_ACTIVE_GAMES) {
		int j;
		for(j = 0; j < PLAYOUTS; j++) {
			game active[MAX_ACTIVE_GAMES];
			int j;
			for(j = 0; j < MAX_ACTIVE_GAMES; j++) {
				int index = i+j;
				if(index < gameCount) {
					active[j] = openGames[index];
				}
			}
			// copy active games to device
			cudaMalloc((void **)&games_d, MAX_ACTIVE_GAMES* sizeof(game));
			cudaMemcpy(games_d, active, MAX_ACTIVE_GAMES*sizeof(game), cudaMemcpyHostToDevice);
			// performs playout
			playouts_d<<<dimGrid, dimBlock>>>(b_d,games_d,rng);
			cudaThreadSynchronize();

			// get result back
			cudaMemcpy(active, games_d, MAX_ACTIVE_GAMES*sizeof(game), cudaMemcpyDeviceToHost);

			// save result
			for(j = 0; j < MAX_ACTIVE_GAMES; j++) {
				int index = i+j;
				if(index < gameCount) {
					openGames[index] = active[j];
				}
			}
			cudaThreadSynchronize();
		}
	}

	// clean up
	cudaFree(b_d);
	cudaFree(rng);
	cudaFree(games_d);

	// report errors
	printf("clean up: %s\n",cudaGetErrorString(cudaGetLastError()));
	// result
	printf("--- Games 1 %d ---\n",PLAYOUTS);
	int k;
	for(k = 0; k < gameCount; k++) {
		game cg = openGames[k];
		printf("%d move %d: win %d, loss %d, draw %d\n",k,cg.move,cg.win,cg.loss,cg.draw);
	}
	printf("--- End Games 1 ---\n");	
}

/* kernel runner that copies always MAX_ACTIVE_GAMES games to device before running kernel */
void rateGame2(const small_board * b) {
	game emptyGame;
	emptyGame.move = -1;
	curandState *rng;
	game *games_d;
	small_board *b_d;
	// prepare kernel call
	dim3 dimBlock( 1, 1 );
	dim3 dimGrid( MAX_ACTIVE_GAMES, 1 );
	// init rng
	cudaMalloc((void **)&rng, MAX_ACTIVE_GAMES* sizeof(curandState));
	setup_rnd_kernel<<<dimBlock, dimGrid>>>(rng, time(0));
	cudaThreadSynchronize();
	// get free moves on board
	idx toPlay[ELEMENT_SIZE];
	short lastIndex = init_toplay(toPlay,b);
	int gameCount = lastIndex+1;
	// create a game for each move
	game openGames[ELEMENT_SIZE];
	int i;
	for(i =0; i < gameCount; i++) {
		init_game(&openGames[i],toPlay[i]);
	}
	// copy original board
	cudaMalloc( (void**)&b_d, sizeof(small_board));
	cudaMemcpy( b_d, b, sizeof(small_board), cudaMemcpyHostToDevice);
	// space for active games on device
	cudaMalloc((void **)&games_d, MAX_ACTIVE_GAMES* sizeof(game));
	// always MAX_ACTIVE_GAMES active
	for(i = 0; i < gameCount; i=i+MAX_ACTIVE_GAMES) {
		game active[MAX_ACTIVE_GAMES];
		int j;
		for(j = 0; j < MAX_ACTIVE_GAMES; j++) {
			int index = i+j;
			if(index < gameCount) {
				active[j] = openGames[index];
			}else {
				active[j] = emptyGame;
			}
		}
		// copy active games to device
		cudaMemcpy(games_d, active, MAX_ACTIVE_GAMES*sizeof(game), cudaMemcpyHostToDevice);
		int k;
		for(k = 0; k < PLAYOUTS; k++) {
			// do playouts for MAX_ACTIVE_GAMES
			playouts_d<<<dimGrid, dimBlock>>>(b_d,games_d,rng);
			cudaThreadSynchronize();
		}
		// copy result back
		cudaMemcpy(active, games_d, MAX_ACTIVE_GAMES*sizeof(game), cudaMemcpyDeviceToHost);

		// save result
		for(j = 0; j < MAX_ACTIVE_GAMES; j++) {
			int index = i+j;
			if(index < gameCount) {
				openGames[index] = active[j];
			}
		}
	}

	// clean up
	cudaFree(games_d);
	cudaFree(b_d);
	cudaFree(rng);

	// report errors
	printf("clean up: %s\n",cudaGetErrorString(cudaGetLastError()));
	// result
	printf("--- Games 2 %d ---\n",PLAYOUTS);
	int k;
	for(k = 0; k < gameCount; k++) {
		game cg = openGames[k];
		printf("%d move %d: win %d, loss %d, draw %d\n",k,cg.move,cg.win,cg.loss,cg.draw);
	}
	printf("--- End Games 2 ---\n");	
}

/* kernel runner that copies all needed data to device before running the kernel */
void rateGame3(const small_board * b) {
	curandState *rng;
	game *games_d;
	small_board *b_d;
	// prepare kernel call
	dim3 dimBlock( 1, 1 );
	dim3 dimGrid( MAX_ACTIVE_GAMES, 1 );
	// init rng
	cudaMalloc((void **)&rng, MAX_ACTIVE_GAMES* sizeof(curandState));
	setup_rnd_kernel<<<dimBlock, dimGrid>>>(rng, time(0));
	cudaThreadSynchronize();
	// get free moves on board
	idx toPlay[ELEMENT_SIZE];
	short lastIndex = init_toplay(toPlay,b);
	int gameCount = lastIndex+1;

	// create a game for each move
	game openGames[ELEMENT_SIZE];
	int i;
	for(i =0; i < gameCount; i++) {
		init_game(&openGames[i],toPlay[i]);
	}
	// copy original board
	cudaMalloc( (void**)&b_d, sizeof(small_board));
	cudaMemcpy( b_d, b, sizeof(small_board), cudaMemcpyHostToDevice);
	
	// space for games on device
	cudaMalloc((void **)&games_d, ELEMENT_SIZE* sizeof(game));
	cudaMemcpy(games_d, openGames, ELEMENT_SIZE*sizeof(game), cudaMemcpyHostToDevice);

	/* always MAX_ACTIVE_GAMES active*/
	for(i = 0; i < gameCount; i=i+MAX_ACTIVE_GAMES) {
		int k;
		for(k = 0; k < PLAYOUTS; k++) {
			// perform playouts
			repeat_playouts_d<<<dimGrid, dimBlock>>>(b_d,games_d,rng,i,gameCount);
			cudaThreadSynchronize();
		}

	}

	cudaMemcpy(openGames, games_d, ELEMENT_SIZE*sizeof(game), cudaMemcpyDeviceToHost);

	// clean up
	cudaFree(games_d);
	cudaFree(b_d);
	cudaFree(rng);

	// report errors
	printf("clean up: %s\n",cudaGetErrorString(cudaGetLastError()));
	// result
	printf("--- Games 3 %d ---\n",PLAYOUTS);
	int k;
	for(k = 0; k < gameCount; k++) {
		game cg = openGames[k];
		printf("%d move %d: win %d, loss %d, draw %d\n",k,cg.move,cg.win,cg.loss,cg.draw);
	}
	printf("--- End Games 3 ---\n");	
}

/* kernel runner that copies always MAX_ACTIVE_GAMES games to device before running kernel uses NO path compression */
int rateGame_npc(const small_board * b) {
	game emptyGame;
	emptyGame.move = -1;
	curandState *rng;
	game *games_d;
	small_board *b_d;
	// prepare kernel call
	dim3 dimBlock( 1, 1 );
	dim3 dimGrid( MAX_ACTIVE_GAMES, 1 );
	// init rng
	cudaMalloc((void **)&rng, MAX_ACTIVE_GAMES * sizeof(curandState));
	setup_rnd_kernel_npc<<<dimBlock, dimGrid>>>(rng, time(0));
	cudaThreadSynchronize();
	// get free moves on board
	idx toPlay[ELEMENT_SIZE];
	short lastIndex = init_toplay(toPlay,b);
	int gameCount = lastIndex+1;
	// create a game for each move
	game openGames[ELEMENT_SIZE];
	int i;
	for(i =0; i < gameCount; i++) {
		init_game(&openGames[i],toPlay[i]);
	}
	// copy original board
	cudaMalloc( (void**)&b_d, sizeof(small_board));
	cudaMemcpy( b_d, b, sizeof(small_board), cudaMemcpyHostToDevice);
	// space for active games on device
	cudaMalloc((void **)&games_d, MAX_ACTIVE_GAMES* sizeof(game));
	// always MAX_ACTIVE_GAMES active
	for(i = 0; i < gameCount; i=i+MAX_ACTIVE_GAMES) {
		game active[MAX_ACTIVE_GAMES];
		int j;
		for(j = 0; j < MAX_ACTIVE_GAMES; j++) {
			int index = i+j;
			if(index < gameCount) {
				active[j] = openGames[index];
			}else {
				active[j] = emptyGame;
			}
		}
		// copy active games to device
		cudaMemcpy(games_d, active, MAX_ACTIVE_GAMES*sizeof(game), cudaMemcpyHostToDevice);
		int k;
		for(k = 0; k < PLAYOUTS; k++) {
			// do playouts for MAX_ACTIVE_GAMES
			playouts_npc_d<<<dimGrid, dimBlock>>>(b_d,games_d,rng);
			cudaThreadSynchronize();
		}
		// copy result back
		cudaMemcpy(active, games_d, MAX_ACTIVE_GAMES*sizeof(game), cudaMemcpyDeviceToHost);

		// save result
		for(j = 0; j < MAX_ACTIVE_GAMES; j++) {
			int index = i+j;
			if(index < gameCount) {
				openGames[index] = active[j];
			}
		}
	}

	// clean up
	cudaFree(games_d);
	cudaFree(b_d);
	cudaFree(rng);

	// report errors
	printf("clean up: %s\n",cudaGetErrorString(cudaGetLastError()));
	// result
	printf("--- Games npc %d %d ---\n",PLAYOUTS,MAX_ACTIVE_GAMES);
	int max_id = -1;
	int max_win = -1;
	int k;
	for(k = 0; k < gameCount; k++) {
		game cg = openGames[k];
		if(cg.win > max_win) {
			max_win = cg.win;
			max_id = k;
		}
		printf("%d move %d: win %d, loss %d, draw %d\n",k,cg.move,cg.win,cg.loss,cg.draw);
	}
	printf("--- End Games npc ---\n");	
	return max_id;
}

/* No cuda kernel involved. use this for testing against CPU (maybe with another compiler) */
void rateGameCpu(const small_board * org_b) {
	// get free moves on board
	idx toPlay[ELEMENT_SIZE];
	idx lastIndex = init_toplay(toPlay,org_b);
	int gameCount = lastIndex+1;
	// create a game for each move
	game openGames[ELEMENT_SIZE];
	int i;
	for(i =0; i < gameCount; i++) {
		init_game(&openGames[i],toPlay[i]);
	}
	for(i = 0; i < gameCount; i++) {
		// create board code
		code_board_10 code;
		init(&code);
		game * g = &openGames[i];
		int j;
		for(j = 0; j < PLAYOUTS; j++) {
			small_board sb;
			small_board * b = &sb;
			// get copy of orignal board
			copy(org_b,b);
			// perform our move
			perf_move(b,g->move,&code);
			// perform playout
			if(playout(b,&code)) {
				// loss, since we moved first
				loss(g);		
			} else {
				// could be a draw
				if(NONE == reason(b) || b->time == ELEMENT_SIZE) {
					draw(g);
				}else {
					win(g);
				}
			}
		}
	}
	// result
	printf("--- Games ---\n");
	int k;
	for(k = 0; k < gameCount; k++) {
		game cg = openGames[k];
		printf("%d move %d: win %d, loss %d, draw %d\n",k,cg.move,cg.win,cg.loss,cg.draw);
	}
	printf("--- End Games ---\n");	
}