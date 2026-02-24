import ReactECharts from 'echarts-for-react'
import { Box, Chip, Typography, Divider } from '@mui/material'
import LocationOnOutlinedIcon from '@mui/icons-material/LocationOnOutlined'
import { useMemo, useState } from 'react'

type OutbreakType = 'Bacterial' | 'Environmental'

export default function OutbreakTracker() {
  const [type, setType] = useState<OutbreakType>('Bacterial')

  // Example monthly data similar to your screenshot
  const months = useMemo(
    () => ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    []
  )

  const series1 = useMemo(() => [1400, 1800, 2400, 3200, 4100, 5200, 4700, 4300, 4000, 3700, 2100], [])
  const series2 = useMemo(() => [2200, 2500, 2800, 3100, 4500, 5600, 6400, 5200, 4600, 4300, 3900], [])

  const chartOption = useMemo(
    () => ({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(30,30,30,0.95)',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        textStyle: { color: '#fff' },
      },
      legend: {
        top: 16,
        left: 16,
        itemWidth: 14,
        itemHeight: 8,
        textStyle: { color: 'rgba(255,255,255,0.75)' },
        data: ['Series 1', 'Series 2'],
      },
      grid: { top: 72, left: 48, right: 28, bottom: 44 },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: months,
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.15)' } },
        axisLabel: { color: 'rgba(255,255,255,0.55)' },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: { color: 'rgba(255,255,255,0.55)' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)', type: 'dashed' } },
      },
      series: [
        {
          name: 'Series 1',
          type: 'line',
          smooth: true,
          data: series1,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: { width: 3, color: '#a78bfa' },
          itemStyle: { color: '#a78bfa' },
          emphasis: { focus: 'series' },
        },
        {
          name: 'Series 2',
          type: 'line',
          smooth: true,
          data: series2,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: { width: 3, color: '#ff6b7a' },
          itemStyle: { color: '#ff6b7a' },
          emphasis: { focus: 'series' },
        },
      ],
    }),
    [months, series1, series2]
  )

  const gaugeValue = 54
  const casesInArea = 407
  const spreadLevel = 'Moderate'

  const gaugeOption = useMemo(
    () => ({
      backgroundColor: 'transparent',
      series: [
        {
          type: 'gauge',
          startAngle: 180,
          endAngle: 0,
          center: ['50%', '58%'],
          radius: '85%',
          min: 0,
          max: 100,
          splitNumber: 0,
          axisLine: {
            lineStyle: {
              width: 18,
              color: [
                [gaugeValue / 100, '#ff5c5c'],
                [1, 'rgba(255,255,255,0.10)'],
              ],
            },
          },
          pointer: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          detail: {
            show: true,
            offsetCenter: [0, 10],
            formatter: '{value}%',
            color: '#ff5c5c',
            fontSize: 34,
            fontWeight: 700,
          },
          title: { show: false },
          data: [{ value: gaugeValue }],
        },
      ],
    }),
    [gaugeValue]
  )

  return (
    <Box
      sx={{
        width: '100vw',
        minHeight: '100vh',
        background: 'radial-gradient(circle at bottom right, #1b2335 0%, #0b0f19 60%)',
        pt: 12, // space below fixed header
        pb: 6,
        px: { xs: 2, md: 6 },
        color: '#fff',
      }}
    >
      {/* Top right location like screenshot */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'rgba(255,255,255,0.85)' }}>
          <Typography sx={{ fontSize: '1.25rem', fontWeight: 600 }}>Houston, TX</Typography>
          <LocationOnOutlinedIcon />
        </Box>
      </Box>

      <Divider sx={{ borderColor: 'rgba(255,92,92,0.55)', mb: 4 }} />

      {/* Main two-column layout */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: '1.2fr 0.8fr' },
          gap: 4,
          alignItems: 'start',
        }}
      >
        {/* LEFT: Chart panel */}
        <Box
          sx={{
            borderRadius: '14px',
            border: '1px solid rgba(255,255,255,0.10)',
            backgroundColor: 'rgba(0,0,0,0.35)',
            overflow: 'hidden',
            boxShadow: '0 16px 40px rgba(0,0,0,0.45)',
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 2.5 }}>
            <Box>
              <Typography sx={{ fontWeight: 700, fontSize: '1.05rem' }}>FLARE Outbreak Tracker</Typography>
              <Typography sx={{ color: 'rgba(255,255,255,0.70)', fontSize: '0.95rem' }}>
                Epidemiology Monitor Chart
              </Typography>
            </Box>

            <Chip
              label="Example chart"
              size="small"
              sx={{
                backgroundColor: 'rgba(255,255,255,0.08)',
                color: 'rgba(255,255,255,0.75)',
              }}
            />
          </Box>

          <Box sx={{ height: 520, px: 1.5, pb: 2 }}>
            <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} />
          </Box>
        </Box>

        {/* RIGHT: Stats panel with dashed border */}
        <Box
          sx={{
            borderRadius: '18px',
            border: '2px dashed rgba(255,255,255,0.25)',
            p: 4,
            backgroundColor: 'rgba(0,0,0,0.25)',
          }}
        >
          <Typography sx={{ color: '#ff5c5c', fontSize: '2.4rem', fontWeight: 800, lineHeight: 1 }}>
            {spreadLevel}
          </Typography>
          <Typography sx={{ color: 'rgba(255,255,255,0.85)', mb: 3, fontWeight: 600 }}>
            Disease Spread Level
          </Typography>

          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3, alignItems: 'center' }}>
            <Box>
              <ReactECharts option={gaugeOption} style={{ height: 220, width: '100%' }} />
              <Typography sx={{ color: 'rgba(255,255,255,0.75)', textAlign: 'center', mt: -1 }}>
                of average case level
              </Typography>
            </Box>

            <Box sx={{ textAlign: 'center' }}>
              <Typography sx={{ color: '#ff5c5c', fontSize: '3.2rem', fontWeight: 800 }}>
                {casesInArea}
              </Typography>
              <Typography sx={{ color: 'rgba(255,255,255,0.75)' }}>
                current cancer patient in
                <br />
                the area
              </Typography>
            </Box>
          </Box>

          {/* Heat map placeholder (matches screenshot area) */}
          <Box
            sx={{
              mt: 4,
              borderRadius: 2,
              overflow: 'hidden',
              border: '1px solid rgba(255,255,255,0.10)',
              backgroundColor: 'rgba(255,255,255,0.04)',
            }}
          >
            <Box
              component="img"
              src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/OpenStreetMap_Logo.svg/512px-OpenStreetMap_Logo.svg.png"
              alt="Regional Epidemiology Heat Map"
              sx={{
                width: '100%',
                height: 220,
                objectFit: 'cover',
                opacity: 0.75,
              }}
            />
          </Box>

          <Typography sx={{ color: 'rgba(255,255,255,0.75)', textAlign: 'center', mt: 1.5 }}>
            Regional Epidemiology Heat Map
          </Typography>
        </Box>
      </Box>

      {/* Bottom outbreak types */}
      <Box sx={{ mt: 5, textAlign: 'center' }}>
        <Typography sx={{ color: '#ff5c5c', fontWeight: 800, fontSize: '1.6rem', mb: 1 }}>
          Outbreak Types
        </Typography>

        <Box sx={{ display: 'inline-flex', gap: 1.5, alignItems: 'center' }}>
          <Chip
            label="Bacterial"
            onClick={() => setType('Bacterial')}
            variant={type === 'Bacterial' ? 'filled' : 'outlined'}
            sx={{
              color: '#fff',
              borderColor: 'rgba(255,255,255,0.25)',
              backgroundColor: type === 'Bacterial' ? 'rgba(255,92,92,0.18)' : 'transparent',
            }}
          />
          <Typography sx={{ color: 'rgba(255,255,255,0.65)' }}>|</Typography>
          <Chip
            label="Environmental"
            onClick={() => setType('Environmental')}
            variant={type === 'Environmental' ? 'filled' : 'outlined'}
            sx={{
              color: '#fff',
              borderColor: 'rgba(255,255,255,0.25)',
              backgroundColor: type === 'Environmental' ? 'rgba(255,92,92,0.18)' : 'transparent',
            }}
          />
        </Box>
      </Box>
    </Box>
  )
}
