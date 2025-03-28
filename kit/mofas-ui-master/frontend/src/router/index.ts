import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import TheWelcome from '@/components/TheWelcome.vue'

const routes = [
  {
    path: '/',
    name: 'TheWelcome',
    component: TheWelcome,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
