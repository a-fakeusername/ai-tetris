<template>
    <div class="tetris-container">
      <h2>Vue Tetris Client</h2>
      <p>Status: {{ connectionStatus }}</p>
      <canvas
        ref="gameCanvas"
        :width="canvasWidth"
        :height="canvasHeight"
        class="game-board"
      ></canvas>
      <p v-if="score !== null">Score: {{ score }}</p>
      <p v-if="isGameOver" class="game-over-text">GAME OVER</p>
      <div class="instructions">
        <p>Use Arrow Keys to Play:</p>
        <ul>
          <li>&larr; : Move Left</li>
          <li>&rarr; : Move Right</li>
          <li>&darr; : Move Down</li>
          <li>&uarr; : Rotate</li>
          </ul>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted, onUnmounted, computed, watch } from 'vue';
  import { io } from 'socket.io-client';
  
  // --- Configuration ---
  const BACKEND_URL = 'http://localhost:5000'; // Your Flask server URL
  const BLOCK_SIZE = 30; // Size of each block in pixels
  
  // --- Reactive State ---
  const gameCanvas = ref(null); // Template ref for the canvas
  const ctx = ref(null); // Canvas rendering context
  const isConnected = ref(false);
  const board = ref([]); // 2D array representing the game board
  const currentPiece = ref(null); // Info about the falling piece
  const score = ref(0);
  const isGameOver = ref(false);
  const boardWidth = ref(10); // Default, will be updated from backend
  const boardHeight = ref(20); // Default, will be updated from backend
  
  let socket = null;
  
  // --- Computed Properties ---
  const connectionStatus = computed(() => {
    if (isGameOver.value) return 'Game Over';
    return isConnected.value ? 'Connected' : 'Disconnected';
  });
  
  const canvasWidth = computed(() => boardWidth.value * BLOCK_SIZE);
  const canvasHeight = computed(() => boardHeight.value * BLOCK_SIZE);
  
  // --- Drawing Functions ---
  const colors = [
      null, // 0 index is empty
      '#00FFFF', // I - Cyan
      '#FFFF00', // O - Yellow
      '#800080', // T - Purple
      '#008000', // S - Green
      '#FF0000', // Z - Red
      '#0000FF', // J - Blue
      '#FFA500', // L - Orange
      '#888888'  // Ghost piece color or default
  ];
  
  function getBlockColor(value) {
      // Example: If backend sends numeric index for color/shape type
      if (value > 0 && value < colors.length) {
          return colors[value];
      }
      // Example: If backend sends CSS color string directly
      if (typeof value === 'string' && value.startsWith('#')) {
           return value;
      }
      return '#333'; // Default color for unknown blocks
  }
  
  function drawBlock(x, y, color) {
    if (!ctx.value || !color) return;
    ctx.value.fillStyle = color;
    ctx.value.fillRect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    ctx.value.strokeStyle = '#555'; // Grid lines
    ctx.value.strokeRect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  }
  
  function drawGame() {
    if (!ctx.value) return;
  
    // Clear canvas
    ctx.value.clearRect(0, 0, canvasWidth.value, canvasHeight.value);
  
    // Draw background (optional, can make empty cells visible)
    ctx.value.fillStyle = '#222';
    ctx.value.fillRect(0, 0, canvasWidth.value, canvasHeight.value);
  
    // Draw the frozen blocks on the board
    board.value.forEach((row, y) => {
      row.forEach((value, x) => {
        if (value !== 0) { // Assuming 0 means empty
          drawBlock(x, y, getBlockColor(value));
        }
      });
    });
  
    // Draw the current falling piece
    if (currentPiece.value && currentPiece.value.shape) {
      currentPiece.value.shape.forEach((row, dy) => {
        row.forEach((value, dx) => {
          if (value !== 0) { // Assuming 0 means empty part of shape matrix
            const boardX = currentPiece.value.x + dx;
            const boardY = currentPiece.value.y + dy;
            // Only draw if within board boundaries (optional safety check)
            if (boardX >= 0 && boardX < boardWidth.value && boardY >= 0 && boardY < boardHeight.value) {
               drawBlock(boardX, boardY, currentPiece.value.color || getBlockColor(value)); // Use piece color if available
            }
          }
        });
      });
    }
  }
  
  // --- WebSocket Logic ---
  function setupWebSocket() {
    console.log('Attempting to connect to backend...');
    socket = io(BACKEND_URL, {
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
  
    socket.on('connect', () => {
      console.log('Connected to backend via WebSocket ID:', socket.id);
      isConnected.value = true;
      // Optional: Join a specific game room if your backend requires it
      // socket.emit('join_game');
    });
  
    socket.on('disconnect', (reason) => {
      console.log('Disconnected from backend:', reason);
      isConnected.value = false;
      // Optionally reset game state or show message
      // score.value = 0;
      // board.value = [];
      // currentPiece.value = null;
      // isGameOver.value = true; // Or indicate disconnection
    });
  
    socket.on('connect_error', (error) => {
      console.error('Connection Error:', error);
      isConnected.value = false;
    });
  
    // Listen for game state updates from the backend
    socket.on('game_update', (gameState) => {
      // console.log('Game state update received:', gameState); // Debugging
      board.value = gameState.board || [];
      currentPiece.value = gameState.current_piece || null;
      score.value = gameState.score !== undefined ? gameState.score : score.value;
      isGameOver.value = gameState.is_game_over || false;
      boardWidth.value = gameState.board_width || boardWidth.value;
      boardHeight.value = gameState.board_height || boardHeight.value;
  
      // Trigger redraw only after state is updated
       requestAnimationFrame(drawGame);
    });
  }
  
  function sendAction(action) {
    if (socket && isConnected.value && !isGameOver.value) {
      // console.log('Sending action:', action); // Debugging
      socket.emit('player_action', { action });
    } else {
        console.warn('Cannot send action - Socket not connected or game over.');
    }
  }
  
  // --- Keyboard Input Handling ---
  function handleKeydown(event) {
    if (!isConnected.value || isGameOver.value) {
      return; // Don't handle input if not connected or game is over
    }
  
    let action = null;
    switch (event.key) {
      case 'ArrowLeft':
        action = 'move_left';
        break;
      case 'ArrowRight':
        action = 'move_right';
        break;
      case 'ArrowDown':
        action = 'move_down';
        break;
      case 'x':
      case 'ArrowUp':
        action = 'rotate';
        break;
      case 'z':
        action = 'rotate_reverse'; // Optional: Reverse rotation
        break;
      case 'a':
        action = 'rotate_180'; // Optional: 180-degree rotation
        break;
      case 'c':
        action = 'hold'; // Optional: Hold piece
        break;
      case ' ': // Space bar
        action = 'hard_drop';
        break;
      default:
        return; // Ignore other keys
    }
  
    if (action) {
        event.preventDefault(); // Prevent default browser action (e.g., scrolling)
        sendAction(action);
    }
  }
  
  // --- Lifecycle Hooks ---
  onMounted(() => {
    if (gameCanvas.value) {
      ctx.value = gameCanvas.value.getContext('2d');
      console.log('Canvas context obtained.');
      // Initial draw or waiting message?
      // ctx.value.fillStyle = 'grey';
      // ctx.value.font = '20px Arial';
      // ctx.value.textAlign = 'center';
      // ctx.value.fillText('Connecting...', canvasWidth.value / 2, canvasHeight.value / 2);
    } else {
        console.error("Canvas element not found!");
    }
  
    setupWebSocket();
    window.addEventListener('keydown', handleKeydown);
  });
  
  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown);
    if (socket) {
      console.log('Disconnecting socket...');
      socket.disconnect();
    }
  });
  
  // Optional: Watch canvas size changes to redraw if needed,
  // though computed properties should handle this via :width/:height bindings.
  // watch([canvasWidth, canvasHeight], () => {
  //   if (ctx.value) {
  //       console.log("Canvas size changed, redrawing...");
  //       requestAnimationFrame(drawGame); // Redraw if dimensions change
  //   }
  // });
  
  </script>
  
  <style scoped>
  .tetris-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: sans-serif;
  }
  
  .game-board {
    border: 3px solid #eee;
    background-color: #222; /* Fallback background */
    margin-top: 10px;
    /* Prevent blurry rendering on some displays */
    image-rendering: -moz-crisp-edges;
    image-rendering: -webkit-crisp-edges;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
  }
  
  .game-over-text {
    color: red;
    font-size: 2em;
    font-weight: bold;
    margin-top: 10px;
  }
  
  .instructions {
    margin-top: 20px;
    text-align: left;
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
     background-color: #f9f9f9;
  }
  
  .instructions ul {
      list-style: none;
      padding-left: 0;
  }
   .instructions li {
       margin-bottom: 5px;
   }
  </style>