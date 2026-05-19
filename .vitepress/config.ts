import { defineConfig } from 'vitepress'
import sidebar from './sidebar'

const isProduction = process.env.NODE_ENV === 'production' || process.argv.includes('build')

export default defineConfig({
  title: '从零开始系统学习深度学习',
  description: '深度学习全栈学习路线教程',
  lang: 'zh-CN',
  base: isProduction ? '/hello-deep-learning/' : '/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'Python 基础', link: '/ch01-python基础/01-什么是python' },
      { text: '神经网络', link: '/ch03-神经网络/01-神经网络基础' },
      { text: '反向传播', link: '/ch05-误差反向传播法/01-计算图' },
      { text: 'CNN', link: '/ch07-卷积神经网络/01-全连接层的问题' },
      { text: 'RNN', link: '/ch09-循环神经网络/01-序列数据是什么' },
      { text: '注意力机制', link: '/ch10-注意力机制/01-注意力是什么' },
      { text: 'CV 进阶', link: '/ch11-计算机视觉进阶/01-图像增广' },
      { text: 'NLP', link: '/ch12-自然语言处理/00-嵌入基础' },
    ],

    sidebar,

    search: {
      provider: 'local',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/baxiang/hello-deep-learning' },
    ],

    outline: {
      label: '本页目录',
      level: [2, 3],
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'medium',
        timeStyle: 'short',
      },
    },
  },

  markdown: {
    lineNumbers: true,
  },

  srcExclude: [
    '**/node_modules/**',
    '**/.vitepress/**',
    '**/__pycache__/**',
    '**/.venv/**',
    '**/code/**',
    'docs/**',
    'AGENTS.md',
    'CLAUDE.md',
    '.github/**',
  ],
})
