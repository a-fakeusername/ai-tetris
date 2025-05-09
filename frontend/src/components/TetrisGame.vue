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
    <button @click="sendAction('restart')">Restart</button>
    <br />
    <button @click="changeMode()">Current: {{ mode }}</button>
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
  const DAS = 100; // Delay before auto repeat (in ms)
  const ARR = 10; // Auto Repeat Rate (in ms)
  const SDF = 50; // Soft Drop Speed (in ms)
  
  // --- Reactive State ---
  const gameCanvas = ref(null); // Template ref for the canvas
  const ctx = ref(null); // Canvas rendering context
  const isConnected = ref(false);
  const board = ref([]); // 2D array representing the game board
  const currentPiece = ref(null); // Info about the falling piece
  const pieceQueue = ref([]); // Queue of upcoming pieces
  const score = ref(0);
  const isGameOver = ref(false);
  const boardWidth = ref(10); // Default, will be updated from backend
  const boardHeight = ref(20); // Default, will be updated from backend
  const mode = ref('player'); // State to manage player vs bot

  const activeEvents = ref(new Set()); // Track active events for key handling
  const eventCount = ref({ // Stores counters, makes sure that release before timeout stops DAS
    'move_left': 0,
    'move_right': 0
  });
  const eventIntervals = ref({}); // Stores intervals which are deleted upon keyup
  
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
      pieceQueue.value = gameState.piece_queue || [];
  
      // Trigger redraw only after state is updated
       requestAnimationFrame(drawGame);
    });
  }
  
  function sendAction(action) {
    if (!socket || !isConnected.value) {
      console.error('Socket not initialized!');
      return;
    }
    if (action == 'restart') {
      socket.emit('player_action', { action: 'restart' });
      console.log("Restarting game...");
    } else if (!isGameOver.value) {
      // console.log('Sending action:', action); // Debugging
      if (mode.value == 'bot') {
        console.warn("Bot is playing, no action sent.");
      } else {
        socket.emit('player_action', { action });
      }
    } else {
      console.warn('Invalid input on game over.');
    }
  }

  function changeMode() {
    mode.value = mode.value == 'player' ? 'bot' : 'player';
    console.log("Changing mode to: " + mode.value);
    socket.emit('player_action', { action: 'change_mode' });
  }

  // Takes in event.key and gets respective action
  function getActionFromKey(key) {
    let action = null;
    switch (key) {
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
        action = 'rotate_reverse';
        break;
      case 'a':
        action = 'rotate_180';
        break;
      case ' ':
        action = 'hard_drop';
        break;
    }
    return action;
  }

  function stopInterval(action) {
    if (activeEvents.value.has(action)) {
      activeEvents.value.delete(action);
    }
    if (eventIntervals.value[action]) {
      // console.log('remove das');
      clearInterval(eventIntervals.value[action]);
      delete eventIntervals.value[action];
    }
  }
  
  // --- Keyboard Input Handling ---
  function handleKeydown(event) {
    if (!isConnected.value || isGameOver.value) {
      return; // Don't handle input if not connected or game is over
    }

    if (event.repeat) {
      return; // Ignore repeated key presses
    }
  
    const action = getActionFromKey(event.key); // Map keyCode to action
    if (action == null) {
      return; // Ignore if no action is mapped to the key
    }
  
    if (action) {
      event.preventDefault(); // Prevent default browser action (e.g., scrolling)
      // Handle DAS (Delayed Auto Shift)
      if (action == 'move_left' || action == 'move_right') {
        if (activeEvents.value.has(action)) {
          return; // Ignore if already active
        }
        
        const otherAction = action == 'move_left' ? 'move_right' : 'move_left';
        stopInterval(otherAction);

        const counter = eventCount.value[action]++;
        activeEvents.value.add(action); // Add to active events
        setTimeout(() => {
          if (activeEvents.value.has(action) && !activeEvents.value.has(otherAction) && eventCount.value[action] == counter + 1) {
            eventIntervals.value[action] = setInterval(() => {
              sendAction(action);
            }, 0);
          }
        }, DAS);
        console.log("das");

      } else if (action == 'move_down') {
        if (activeEvents.value.has(action)) {
          return;
        }

        activeEvents.value.add(action); // Add to active events
        eventIntervals.value[action] = setInterval(() => {
          sendAction(action);
        }, SDF);
        console.log("soft drop");
      }
      sendAction(action);
    }
  }
  
  // Used to cancel DAS/Soft Drop
  function handleKeyup(event) {
    if (!isConnected.value || isGameOver.value) {
      return; // Don't handle input if not connected or game is over
    }
    let action = getActionFromKey(event.key);
    if (action == 'move_left' || action == 'move_right' || action == 'move_down') {
      stopInterval(action);
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
    window.addEventListener('keyup', handleKeyup);
  });
  
  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown);
    window.removeEventListener('keyup', handleKeyup);
    if (socket) {
      console.log('Disconnecting socket...');
      socket.disconnect();
    }
  });
  
</script>