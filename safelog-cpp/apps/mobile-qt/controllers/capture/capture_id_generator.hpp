#pragma once

#include "safelog/contracts/ports.hpp"

#include <QUuid>
#include <string>

namespace safelog::qtapp {

class CaptureIdGenerator final : public IIdGenerator {
public:
    Id next(const std::string& prefix) override
    {
        const auto uuid =
            QUuid::createUuid().toString(QUuid::WithoutBraces);

        return prefix + "-" + uuid.toStdString();
    }
};

} // namespace safelog::qtapp