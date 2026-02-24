import { useEffect, useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'
import { fetchPatientHistory } from '../api/patientData'
import type { PatientRecord } from '../api/patientData'
export default function PatientData() {
  const { getAccessTokenSilently } = useAuth0()
  const [records, setRecords] = useState<PatientRecord[]>([])

  useEffect(() => {
    const load = async () => {
      const token = await getAccessTokenSilently()
      const data = await fetchPatientHistory(token)
      setRecords(data)
    }
    load()
  }, [])

  return ( //UI goes in here
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody>
        {records.map((r) => (
          <tr key={r.id}>
            <td>{r.date}</td>
            <td>{r.result}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
