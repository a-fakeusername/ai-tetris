# Stage 1: Build the Vue.js application
FROM node:24-alpine AS builder

WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of your application source code
COPY . .

# Build the application for production
RUN npm run build # This typically creates a 'dist' folder

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

# Copy the built static assets from the 'builder' stage to Nginx's web root
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

# Command to run Nginx
CMD ["nginx", "-g", "daemon off;"]