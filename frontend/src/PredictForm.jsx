import React, { useState } from "react";
import axios from "axios";
import './PredictForm.css';

const PredictForm = () => {
    const [formData, setFormData] = useState({
        accommodates: "1",
        bathrooms: "1",
        bedrooms: "1",
        beds: "1",
        minimum_nights: "1",
        number_of_reviews: "0",
        review_scores_rating: "80",
        instant_bookable: "0",
        neighbourhood_density: "0.01",
        host_experience: "1",
        room_type_Entire_home_apt: "0",
        neighbourhood_encoded: "1",
        amenity_score: "50"
    });
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleChange = (e) => {
        const { name, value } = e.target;
        let processedValue = value;
        
        if (name === "neighbourhood_density") {
            processedValue = value.replace(/[^0-9.]/g, '');
            const parts = processedValue.split('.');
            if (parts.length > 1) {
                processedValue = `${parts[0]}.${parts[1].slice(0, 4)}`;
            }
        } else if (name === "neighbourhood_encoded" || name === "number_of_reviews") {
            processedValue = value.replace(/\D/g, '');
        }
        
        setFormData(prev => ({
            ...prev,
            [name]: processedValue
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setPrediction(null);
        
        try {
            const payload = {
                accommodates: Math.max(1, Number(formData.accommodates)),
                bathrooms: Math.max(0.1, Number(formData.bathrooms)),
                bedrooms: Math.max(1, Number(formData.bedrooms)),
                beds: Math.max(1, Number(formData.beds)),
                minimum_nights: Math.max(1, Number(formData.minimum_nights)),
                number_of_reviews: Math.round(Number(formData.number_of_reviews)),
                review_scores_rating: Math.min(100, Math.max(0, Number(formData.review_scores_rating))),
                instant_bookable: Number(formData.instant_bookable),
                neighbourhood_density: Math.max(0.0001, Number(formData.neighbourhood_density)),
                host_experience: Math.max(0, Number(formData.host_experience)),
                room_type_Entire_home_apt: Number(formData.room_type_Entire_home_apt),
                neighbourhood_encoded: Math.round(Number(formData.neighbourhood_encoded)),
                amenity_score: Math.max(0, Number(formData.amenity_score))
            };

            const response = await axios.post(
                "http://localhost:8000/predict", 
                payload,
                {
                    headers: { 'Content-Type': 'application/json' }
                }
            );
            
            if (!response.data?.predicted_price) {
                throw new Error("Formato de respuesta inválido");
            }
            
            setPrediction(response.data.predicted_price);
        } catch (err) {
            const backendError = err.response?.data?.detail;
            setError(
                backendError 
                    ? `Error de validación: ${Array.isArray(backendError) 
                        ? backendError.map(e => e.msg).join(', ') 
                        : JSON.stringify(backendError)}`
                    : err.message || "Error desconocido"
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="form-container">
            <h2>Calculadora de Precio para Inversores</h2>
            
            <form onSubmit={handleSubmit}>
                {/* Campos numéricos */}
                {["accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights"].map((field) => (
                    <div key={field} className="form-group">
                        <label>
                            {field.replace(/_/g, " ")}:
                            <input
                                type="number"
                                name={field}
                                value={formData[field]}
                                onChange={handleChange}
                                step="1"
                                min="1"
                                required
                            />
                        </label>
                    </div>
                ))}

                <div className="form-group">
                    <label>
                        Number of reviews:
                        <input
                            type="number"
                            name="number_of_reviews"
                            value={formData.number_of_reviews}
                            onChange={handleChange}
                            step="1"
                            min="0"
                            required
                        />
                    </label>
                </div>

                <div className="form-group">
                    <label>
                        Review scores rating (0-100):
                        <input
                            type="number"
                            name="review_scores_rating"
                            value={formData.review_scores_rating}
                            onChange={handleChange}
                            step="1"
                            min="0"
                            max="100"
                            required
                        />
                    </label>
                </div>

                <div className="form-group">
                    <label>
                        Neighbourhood density:
                        <input
                            type="number"
                            name="neighbourhood_density"
                            value={formData.neighbourhood_density}
                            onChange={handleChange}
                            step="0.0001"
                            min="0.0001"
                            required
                            placeholder="Ej: 0.0064"
                        />
                    </label>
                </div>

                <div className="form-group">
                    <label>
                        Host experience:
                        <input
                            type="number"
                            name="host_experience"
                            value={formData.host_experience}
                            onChange={handleChange}
                            step="1"
                            min="0"
                            required
                        />
                    </label>
                </div>

                <div className="form-group">
                    <label>
                        Neighbourhood encoded:
                        <input
                            type="number"
                            name="neighbourhood_encoded"
                            value={formData.neighbourhood_encoded}
                            onChange={handleChange}
                            step="1"
                            min="0"
                            required
                        />
                    </label>
                </div>

                <div className="form-group">
                    <label>
                        Amenity score:
                        <input
                            type="number"
                            name="amenity_score"
                            value={formData.amenity_score}
                            onChange={handleChange}
                            step="1"
                            min="0"
                            required
                        />
                    </label>
                </div>

                {/* Campos booleanos */}
                <div className="form-group">
                    <label>
                        Instant Bookable:
                        <select
                            name="instant_bookable"
                            value={formData.instant_bookable}
                            onChange={handleChange}
                            required
                        >
                            <option value="0">No</option>
                            <option value="1">Sí</option>
                        </select>
                    </label>
                </div>

                <div className="form-group">
                    <label>
                        Tipo de propiedad:
                        <select
                            name="room_type_Entire_home_apt"
                            value={formData.room_type_Entire_home_apt}
                            onChange={handleChange}
                            required
                        >
                            <option value="0">Habitación</option>
                            <option value="1">Casa completa</option>
                        </select>
                    </label>
                </div>

                <button type="submit" disabled={loading}>
                    {loading ? "Calculando..." : "Predecir Precio"}
                </button>
            </form>

            {error && (
                <div className="error">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {prediction !== null && (
                <div className="result">
                    <h3>Precio estimado:</h3>
                    <p>€{prediction.toLocaleString("es-ES", { 
                        minimumFractionDigits: 2, 
                        maximumFractionDigits: 2 
                    })}</p>
                </div>
            )}
        </div>
    );
};

export default PredictForm;