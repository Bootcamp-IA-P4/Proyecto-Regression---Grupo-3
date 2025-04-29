// frontend/src/PredictForm.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";

const PredictForm = () => {
    const [formData, setFormData] = useState({
        accommodates: "",
        bathrooms: "",
        bedrooms: "",
        beds: "",
        minimum_nights: "",
        number_of_reviews: "",
        review_scores_rating: "",
        instant_bookable: "0",
        neighbourhood_density: "",
        host_experience: "",
        room_type_Entire_home_apt: "0",
        neighbourhood_encoded: "",
        amenity_score: "",
    });
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        
        try {
            // Convertir números a tipo correcto
            const payload = Object.fromEntries(
                Object.entries(formData).map(([key, val]) => [
                    key, 
                    isNaN(val) ? val : Number(val)
                ])
            );

            const response = await axios.post(
                "http://localhost:8000/predict", 
                payload
            );
            
            setPrediction(response.data.predicted_price);
        } catch (err) {
            setError(err.response?.data?.detail || "Error desconocido");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="form-container">
            <h2>Calculadora de Precio para Inversores</h2>
            
            <form onSubmit={handleSubmit}>
                {/* Campos numéricos */}
                {["accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights", 
                  "number_of_reviews", "review_scores_rating", "neighbourhood_density",
                  "host_experience", "neighbourhood_encoded", "amenity_score"].map((field) => (
                    <div key={field} className="form-group">
                        <label>
                            {field.replace(/_/g, " ")}:
                            <input
                                type="number"
                                name={field}
                                value={formData[field]}
                                onChange={handleChange}
                                step={field.includes("review") ? "1" : "0.1"}
                                required
                            />
                        </label>
                    </div>
                ))}

                {/* Campos booleanos (select) */}
                <div className="form-group">
                    <label>
                        Instant Bookable:
                        <select
                            name="instant_bookable"
                            value={formData.instant_bookable}
                            onChange={handleChange}
                        >
                            <option value="0">No</option>
                            <option value="1">Sí</option>
                        </select>
                    </label>
                </div>

                <div className="form-group">
                    <label>
                        Tipo de propiedad (Casa completa):
                        <select
                            name="room_type_Entire_home_apt"
                            value={formData.room_type_Entire_home_apt}
                            onChange={handleChange}
                        >
                            <option value="0">No</option>
                            <option value="1">Sí</option>
                        </select>
                    </label>
                </div>

                <button type="submit" disabled={loading}>
                    {loading ? "Calculando..." : "Predecir Precio"}
                </button>
            </form>

            {error && <div className="error">{error}</div>}
            {prediction && (
                <div className="result">
                    <h3>Precio estimado:</h3>
                    <p>€{prediction.toLocaleString("es-ES")}</p>
                </div>
            )}
        </div>
    );
};

export default PredictForm;