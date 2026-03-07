FROM node:18-alpine as build
WORKDIR /usr/src/app
# copy package files from build context (context is ./frontend)
COPY package*.json ./
RUN npm ci --silent
COPY . ./
RUN npm run build

FROM nginx:stable-alpine
RUN apk add --no-cache gettext
COPY --from=build /usr/src/app/dist /usr/share/nginx/html
# copy entrypoint that writes /usr/share/nginx/html/env.js from env vars
COPY docker-entrypoint.sh /docker-entrypoint.d/docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.d/docker-entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/docker-entrypoint.d/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
e