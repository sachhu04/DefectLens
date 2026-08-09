export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Space Grotesk"', 'sans-serif'],
      },
      colors: {
        dark: {
          bg: '#09090b', // zinc-950
          card: '#18181b', // zinc-900
          border: '#27272a', // zinc-800
          text: '#fafafa', // zinc-50
          muted: '#a1a1aa' // zinc-400
        },
        primary: {
          DEFAULT: '#3b82f6', // blue-500
          hover: '#2563eb' // blue-600
        },
        status: {
          normal: '#10b981', // emerald-500
          anomaly: '#ef4444', // red-500
          warning: '#f59e0b' // amber-500
        }
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.4s ease-out'
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        }
      }
    },
  },
  plugins: [],
}
